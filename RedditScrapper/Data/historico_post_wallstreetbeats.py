import json
from datetime import datetime, UTC

archivo_entrada = r"D:\Descargas\reddit\subreddits25\EducatedInvesting_submissions\EducatedInvesting_submissions"
archivo_salida = "EducatedInvesting_2023_2025.json"

fecha_inicio = datetime(2023,1,1,tzinfo=UTC).timestamp()

guardados = 0
leidos = 0

with open(archivo_entrada, "r", encoding="utf-8", errors="ignore") as f, open(archivo_salida, "w", encoding="utf-8") as out:
    
    for linea in f:
        leidos += 1
        try:
            post = json.loads(linea)
            created = post.get("created_utc")

            if created and float(created) >= fecha_inicio:
                out.write(json.dumps(post, ensure_ascii=False) + "\n")
                guardados += 1

        except:
            pass

print("Posts leídos:", leidos)
print("Posts guardados:", guardados)