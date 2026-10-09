#!/usr/bin/env bash
# Removes the testbed (deleting the namespaces deletes all veth pairs in them).
ip netns del cli 2>/dev/null || true
ip netns del srv 2>/dev/null || true
echo "testbed removed"
