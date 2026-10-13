"""Prueba extremo a extremo contra la API en ejecución: python eval/e2e.py eval/questions.X.jsonl [salida.json]"""
import json, sys, time, urllib.request

API = "http://127.0.0.1:8000"


def post(path, body=None):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else b"", method="POST",
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


qs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
cid = post("/conversations")["id"]
out = []
for q in qs:
    t = time.time()
    m = post(f"/conversations/{cid}/messages", {"question": q["question"]})
    pages = sorted({c["page_start"] for c in m["citations"]})
    ok_cite = bool(set(pages) & set(q["pages"])) if q.get("answerable", True) else None
    out.append({"id": q["id"], "question": q["question"], "answerable": q.get("answerable", True), "status": m["status"],
                "answer": m["content"], "cited_pages": pages, "expected_pages": q["pages"], "cite_hits_expected": ok_cite,
                "seconds": round(time.time() - t, 1)})
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "eval/_e2e.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
ans = [o for o in out if o["answerable"]]; un = [o for o in out if not o["answerable"]]
print(f"con respuesta: {len(ans)}  respondidas(ok)={sum(o['status']=='ok' for o in ans)}  abstenidas={sum(o['status']=='abstained' for o in ans)}  "
      f"cita en página esperada={sum(bool(o['cite_hits_expected']) for o in ans)}  sin verificar={sum(o['status']=='unverified' for o in ans)}")
print(f"sin respuesta: {len(un)}  abstenidas correctas={sum(o['status']=='abstained' for o in un)}  respondidas (alucinación)={sum(o['status']!='abstained' for o in un)}")
