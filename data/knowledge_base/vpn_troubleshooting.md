# VPN Troubleshooting Guide

## Symptoms
- VPN disconnects every 10–15 minutes
- Unable to connect to VPN
- VPN connected but cannot access internal resources

## Common Causes and Fixes

### Frequent Disconnections
1. Check your internet connection stability — VPN requires a stable base connection.
2. Switch to a wired connection if on Wi-Fi.
3. Update the VPN client to the latest version (current: v4.2.1).
4. Change VPN server region: go to Settings → Server → select "Auto".
5. Disable battery saver / power management for the VPN process.

### Cannot Connect
1. Verify your credentials are not expired — check with Okta.
2. Ensure you are not already connected on another device.
3. Restart the VPN client and try again.
4. If on a corporate network, VPN may not be required — try disconnecting.

### Connected but No Internal Access
1. Check if the issue is with a specific service or all internal tools.
2. Run: `ping internal.company.com` — if this fails, escalate to IT.
3. Clear DNS cache: `sudo dscacheutil -flushcache` (macOS).

## Escalation Criteria
Escalate to IT if:
- Issue persists after all steps above
- Multiple teammates are affected simultaneously
- VPN client shows "Authentication failure" after credential reset
