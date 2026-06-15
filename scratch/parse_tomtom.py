import re
import json

try:
    with open("docs/phase2_visualizations/Tomtom Results.html", encoding="utf-8") as f:
        html = f.read()

    # MEME suite usually stores its data in a var data = {...}; block
    m = re.search(r"var\s+data\s*=\s*(\{.*?\});", html, re.DOTALL)
    if not m:
        print("Could not find 'var data = ' in HTML.")
    else:
        data = json.loads(m.group(1))
        targets = {t["id"]: t.get("alt", t["id"]) for t in data.get("targets", [])}

        print("Top Motif Matches:")
        all_matches_flat = []
        for q_match in data.get("all_matches", []):
            q_idx = q_match["idx"]
            q_id = data["queries"][q_idx]["id"]
            for t_match in q_match.get("matches", []):
                t_idx = t_match["idx"]
                t_id = data["targets"][t_idx]["id"]
                t_alt = data["targets"][t_idx].get("alt", t_id)
                evalue = float(t_match["ev"])
                all_matches_flat.append((q_id, t_alt, evalue))

        all_matches_flat.sort(key=lambda x: x[2])
        for q, t, e in all_matches_flat[:15]:
            print(f"Filter: {q} matched with Transcription Factor: {t} (E-value: {e})")

except Exception as e:
    print(f"Error parsing HTML: {e}")
