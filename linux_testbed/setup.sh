#!/usr/bin/env bash
# Real-packet emulation testbed (Linux only, needs root and the sch_netem module).
# Creates two network namespaces joined by one veth pair PER TECHNOLOGY:
#
#   [ns cli] c-wired ---- s-wired [ns srv]   10.0.1.0/24
#            c-wifi  ---- s-wifi             10.0.2.0/24
#            c-5g    ---- s-5g               10.0.3.0/24
#            c-leo   ---- s-leo              10.0.4.0/24
#            c-geo   ---- s-geo              10.0.5.0/24
#
# replay_trace.py then drives delay/loss/rate on each pair from the simulator's
# timeline, so real apps (iperf3, ffmpeg, a WebRTC call, MPTCP, QUIC) feel the trip.
set -euo pipefail
LINKS=(wired wifi 5g leo geo)

ip netns add cli 2>/dev/null || true
ip netns add srv 2>/dev/null || true
for i in "${!LINKS[@]}"; do
  n=${LINKS[$i]}; net=$((i + 1))
  ip link add "c-$n" type veth peer name "s-$n"
  ip link set "c-$n" netns cli
  ip link set "s-$n" netns srv
  ip -n cli addr add "10.0.$net.1/24" dev "c-$n"
  ip -n srv addr add "10.0.$net.2/24" dev "s-$n"
  ip -n cli link set "c-$n" up
  ip -n srv link set "s-$n" up
  # one routing table per interface so each source address leaves on its own link
  ip -n cli rule add from "10.0.$net.1" table "$((100 + net))"
  ip -n cli route add "10.0.$net.0/24" dev "c-$n" scope link table "$((100 + net))"
  ip -n cli route add default via "10.0.$net.2" dev "c-$n" table "$((100 + net))"
done
ip -n cli link set lo up
ip -n srv link set lo up

# MPTCP (kernel >= 5.6): allow up to 8 subflows, every client interface is a subflow endpoint
for ns in cli srv; do
  ip netns exec $ns sysctl -qw net.mptcp.enabled=1 || echo "MPTCP not available in this kernel"
  ip -n $ns mptcp limits set subflow 8 add_addr_accepted 8 2>/dev/null || true
done
for i in "${!LINKS[@]}"; do
  ip -n cli mptcp endpoint add "10.0.$((i + 1)).1" dev "c-${LINKS[$i]}" subflow 2>/dev/null || true
done

# netem on the client side of every pair (egress). Server side mirrors it for RTT.
if ! ip netns exec cli tc qdisc add dev c-wifi root netem delay 1ms 2>/dev/null; then
  echo "WARNING: netem not available (try: sudo modprobe sch_netem). Links work, but can't be impaired."
else
  ip netns exec cli tc qdisc del dev c-wifi root
fi
echo "testbed up. try:  sudo ip netns exec cli ping -c3 10.0.2.2"
