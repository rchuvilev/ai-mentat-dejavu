# Deploy

```sh
npm run build
mkdir -p site/dist && cp dist/*.js site/dist/
# drop server-only modules from the browser bundle
rm -f site/dist/{server,cli,test,test_actions,ui}.js*

python3 deploy/pages_deploy.py ai-dejavu <site-root>
```

Always verify over the wire — never trust the upload response:

```sh
curl -s -o /dev/null -w '%{http_code}\n' https://ai-dejavu.hexstack.app/ai-dejavu/
```

## Live

- https://hexstack.app/ai-dejavu/            (apex, primary)
- https://ai-dejavu.hexstack.app/ai-dejavu/  (subdomain)
- https://ai-dejavu.pages.dev/ai-dejavu/     (Pages subdomain)

## Notes

Cloudflare Pages project `ai-dejavu`, account `6dda101664eca020bf086a4de83118a6`,
zone `hexstack.app` = `2aa5defa47b368ac81d43590dada5004`.

Two tokens, NOT interchangeable: `CLOUDFLARE_API_TOKEN` for Pages,
`CLOUDFLARE_ZONES_TOKEN` for DNS.

### Apex DNS: Error 1000 and the fix

The apex originally had 2 A + 2 AAAA records pointing at **Cloudflare's own proxy
IPs** (104.21.10.18, 172.67.162.30, 2606:4700:3037::ac43:a21e,
2606:4700:3032::6815:a12). That is self-referential, so Cloudflare refused to
serve it with **Error 1000 "DNS points to prohibited IP"** — not a cert delay, and
no amount of waiting fixes it.

Fix: delete those 4 records, add `CNAME @ -> ai-dejavu.pages.dev` (proxied).
The apex began serving 200 within ~90 s, while the Pages domain status still read
`pending` — status lags reality, so verify over the wire.

Both `google-site-verification` TXT records were left intact.
`wwwaaagh.hexstack.app` (CNAME to its own Pages project) was unaffected.
`n8n.` and `mc.` still 403 — they carry the same dead-origin A records and were
already broken before this change; left alone deliberately.

Full DNS backup before the change:
`/var/minis/shared/webhost/dns-backup/hexstack-app-*.json`
