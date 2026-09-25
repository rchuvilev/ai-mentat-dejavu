#!/usr/bin/env python3
"""Cloudflare Pages direct upload without wrangler.

Usage: pages_deploy.py <project> <root-dir>

Auth: CLOUDFLARE_API_TOKEN (Pages scope). DNS needs CLOUDFLARE_ZONES_TOKEN
(a different token) but this script does not touch DNS.

The undocumented bit is the asset hash: blake2b over
(base64(content) + extension-without-dot), digest_size=16 -> 32 hex chars.
Steps 1 and 6 use the ACCOUNT token; steps 3/4/5 use the upload JWT.
"""
import base64
import hashlib
import json
import mimetypes
import os
import subprocess
import sys
import tempfile

ACCOUNT = "6dda101664eca020bf086a4de83118a6"
API = "https://api.cloudflare.com/client/v4"


def curl(args, token):
    cmd = ["curl", "-s", "-H", "Authorization: Bearer " + token] + args
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    try:
        return json.loads(out)
    except Exception:
        print("non-JSON response:", out[:500])
        raise


def die(msg, payload=None):
    print("ERROR:", msg)
    if payload is not None:
        print(json.dumps(payload)[:800])
    sys.exit(1)


def main():
    if len(sys.argv) < 3:
        die("usage: pages_deploy.py <project> <root-dir>")
    project, root = sys.argv[1], os.path.abspath(sys.argv[2])
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token:
        die("CLOUDFLARE_API_TOKEN not set")
    if not os.path.isdir(root):
        die("not a directory: " + root)

    # ---- collect files ------------------------------------------------
    files = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            files.append((rel, full))
    if not files:
        die("no files under " + root)

    print("collecting %d file(s) from %s" % (len(files), root))

    manifest = {}
    blobs = {}
    for rel, full in sorted(files):
        content = open(full, "rb").read()
        b64 = base64.b64encode(content).decode()
        ext = os.path.splitext(rel)[1].lstrip(".")
        h = hashlib.blake2b((b64 + ext).encode(), digest_size=16).hexdigest()
        manifest["/" + rel] = h
        ctype = mimetypes.guess_type(rel)[0] or "application/octet-stream"
        blobs[h] = {"key": h, "value": b64,
                    "metadata": {"contentType": ctype}, "base64": True}
        print("  %-46s %s  %6d B" % (rel, h[:12] + "...", len(content)))

    # ---- 1. upload token ---------------------------------------------
    r = curl(["%s/accounts/%s/pages/projects/%s/upload-token"
              % (API, ACCOUNT, project)], token)
    if not r.get("success"):
        die("upload-token failed", r)
    jwt = r["result"]["jwt"]
    print("got upload JWT")

    # ---- 3. which hashes are missing ---------------------------------
    hashes = sorted(set(manifest.values()))
    r = curl(["-X", "POST", "%s/pages/assets/check-missing" % API,
              "-H", "Content-Type: application/json",
              "--data", json.dumps({"hashes": hashes})], jwt)
    if not r.get("success"):
        die("check-missing failed", r)
    missing = r["result"]
    print("%d/%d asset(s) need upload" % (len(missing), len(hashes)))

    # ---- 4. upload in batches ----------------------------------------
    if missing:
        batch = [blobs[h] for h in missing if h in blobs]
        for i in range(0, len(batch), 40):
            chunk = batch[i:i + 40]
            with tempfile.NamedTemporaryFile("w", suffix=".json",
                                             delete=False) as tf:
                json.dump(chunk, tf)
                tmp = tf.name
            r = curl(["-X", "POST", "%s/pages/assets/upload" % API,
                      "-H", "Content-Type: application/json",
                      "--data", "@" + tmp], jwt)
            os.unlink(tmp)
            if not r.get("success"):
                die("upload failed", r)
            print("  uploaded %d asset(s)" % len(chunk))

    # ---- 5. upsert hashes --------------------------------------------
    r = curl(["-X", "POST", "%s/pages/assets/upsert-hashes" % API,
              "-H", "Content-Type: application/json",
              "--data", json.dumps({"hashes": hashes})], jwt)
    if not r.get("success"):
        print("  warn: upsert-hashes:", json.dumps(r)[:200])

    # ---- 6. create deployment (multipart, ACCOUNT token) -------------
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(manifest, tf)
        mpath = tf.name
    r = curl(["-X", "POST",
              "%s/accounts/%s/pages/projects/%s/deployments"
              % (API, ACCOUNT, project),
              "-F", "manifest=<" + mpath,
              "-F", "branch=main"], token)
    os.unlink(mpath)
    if not r.get("success"):
        die("deployment failed", r)

    res = r["result"]
    print("\ndeployed: %s" % res.get("url"))
    print("id: %s" % res.get("id"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
