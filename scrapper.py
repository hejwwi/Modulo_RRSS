#!/usr/bin/env python3
"""
NVIDIA Finance Reddit Scraper — Enero 2026 → Hoy
=================================================
Scrapea los subreddits financieros más importantes buscando posts
relacionados con NVIDIA. Para cada día devuelve los N posts más
relevantes (por score) y descarga sus imágenes con nombre:
    YYYY-MM-DD_<post_id>.<ext>

Subreddits financieros monitorizados:
    wallstreetbets, stocks, investing, StockMarket, options,
    ValueInvesting, SecurityAnalysis, finance, algotrading,
    Daytrading, dividends, Economics, financialindependence

Uso:
    python nvidia_finance_scraper.py                    # enero 2026 → hoy
    python nvidia_finance_scraper.py --top 5            # top 5 posts/día
    python nvidia_finance_scraper.py --from 2026-02-01
    python nvidia_finance_scraper.py --no-images        # sin descargar imgs
    python nvidia_finance_scraper.py --window 1         # ventanas de 1 día (más exacto)
    python nvidia_finance_scraper.py --query 0 5        # solo algunas queries

Dependencias: pip install requests
"""

import json, time, random, argparse, sys, os, csv, re, urllib.request
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Optional
from collections import defaultdict

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("[ERROR] pip install requests"); sys.exit(1)


# ─── ANSI ─────────────────────────────────────────────────────────────────────

NO_COLOR = not sys.stdout.isatty() or bool(os.environ.get("NO_COLOR"))
def col(t, c=""): return f"{c}{t}\033[0m" if (c and not NO_COLOR) else str(t)
G="\033[92m"; Y="\033[93m"; R="\033[91m"; CY="\033[96m"
GR="\033[90m"; W="\033[97m"; B="\033[1m"; M="\033[95m"

def ok(m):    print(col(f"  ✓ {m}", G))
def warn(m):  print(col(f"  ⚠ {m}", Y))
def err(m):   print(col(f"  ✗ {m}", R))
def info(m):  print(col(f"  · {m}", GR))
def step(m):  print(col(f"\n► {m}", W+B))
def hdr(m):
    print(col(f"\n{'═'*66}", CY))
    print(col(f"  {m}", CY+B))
    print(col(f"{'═'*66}", CY))
def section(m): print(col(f"\n  ── {m}", M))

def pbar(done, total, w=32):
    p = done / total if total else 0
    f = int(w * p)
    return col(f"[{'█'*f}{'░'*(w-f)}] {p*100:.0f}%", GR)


# ─── Subreddits financieros ───────────────────────────────────────────────────

FINANCE_SUBREDDITS = [
    "wallstreetbets",       # el más activo en NVDA options y memes
    "stocks",               # análisis fundamentales y noticias
    "investing",            # inversión a largo plazo
    "StockMarket",          # noticias de mercado
    "options",              # opciones sobre NVDA
    "ValueInvesting",       # análisis de valor
    "SecurityAnalysis",     # análisis técnico profundo
    "finance",              # finanzas generales
    "algotrading",          # trading algorítmico (NVDA en IA/quant)
    "Daytrading",           # trading intradía
    "dividends",            # NVDA dividendos
    "Economics",            # macro que afecta a semis
    "financialindependence",# FIRE community con posiciones NVDA
]


# ─── Queries NVIDIA ───────────────────────────────────────────────────────────

SEARCH_QUERIES = [
    {"id": 0, "label": "NVDA / NVIDIA General",
     "q": 'NVDA OR "$NVDA" OR NVIDIA'},
    {"id": 1, "label": "Earnings / Revenue / Guidance",
     "q": '(earnings OR revenue OR guidance OR EPS OR "beat estimates") NVDA OR NVIDIA'},
    {"id": 2, "label": "Blackwell / H100 / AI chips",
     "q": 'Blackwell OR "H100" OR "B200" OR "GB200" OR "AI chips" NVIDIA'},
    {"id": 3, "label": "Jensen Huang / CEO",
     "q": '"Jensen Huang" OR "Jensen" NVIDIA CEO'},
    {"id": 4, "label": "Price target / analyst",
     "q": '(price target OR analyst OR upgrade OR downgrade OR "buy rating") NVDA OR NVIDIA'},
    {"id": 5, "label": "Short / puts / options",
     "q": '(short OR puts OR calls OR options OR "short squeeze") NVDA OR "$NVDA"'},
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) "
    "Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
]


# ─── Helpers HTTP ─────────────────────────────────────────────────────────────

def build_session():
    s = requests.Session()
    r = Retry(total=5, backoff_factor=2, status_forcelist=[500,502,503,504],
              allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s

def ua(): return random.choice(USER_AGENTS)
def jitter(ms): return (ms + random.uniform(0, ms * 0.65)) / 1000.0
def sleep_j(ms, lbl=""):
    s = jitter(ms)
    if lbl: info(f"{lbl}: {s:.1f}s")
    else: time.sleep(s); return
    time.sleep(s)


# ─── Extracción de imagen ─────────────────────────────────────────────────────

def extract_image(d: dict) -> Optional[str]:
    """Cascada de fallback para obtener la mejor URL de imagen."""
    # URL directa
    if d.get("url") and re.search(r"\.(jpg|jpeg|png|gif|webp)(\?|$)", d["url"], re.I):
        return d["url"]
    # Reddit preview alta resolución
    try:
        return d["preview"]["images"][0]["source"]["url"].replace("&amp;","&")
    except Exception: pass
    # Reddit preview resoluciones
    try:
        res = d["preview"]["images"][0]["resolutions"]
        if res: return res[-1]["url"].replace("&amp;","&")
    except Exception: pass
    # Gallery
    if d.get("is_gallery") and d.get("media_metadata"):
        try:
            k = list(d["media_metadata"].keys())[0]
            m = d["media_metadata"][k]
            if m.get("p"): return m["p"][-1]["u"].replace("&amp;","&")
            if m.get("s",{}).get("u"): return m["s"]["u"].replace("&amp;","&")
        except Exception: pass
    # Thumbnail último recurso
    t = d.get("thumbnail","")
    if t and t not in ("self","default","nsfw","spoiler","") and t.startswith("http"):
        return t
    return None

def img_extension(url: str) -> str:
    m = re.search(r"\.(jpg|jpeg|png|gif|webp)", url, re.I)
    return m.group(1).lower() if m else "jpg"

def download_image(url: str, filepath: str, session) -> bool:
    """Descarga imagen. Devuelve True si éxito."""
    try:
        r = session.get(url, headers={"User-Agent": ua()}, timeout=15, stream=True)
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        return False


# ─── Construcción de post ─────────────────────────────────────────────────────

def make_post(raw: dict, query: dict, subreddit: str) -> dict:
    img = extract_image(raw)
    created = raw.get("created_utc", 0)
    dt = datetime.fromtimestamp(created, tz=timezone.utc)
    return {
        "id":            raw.get("id",""),
        "date":          dt.strftime("%Y-%m-%d"),
        "datetime_iso":  dt.isoformat(),
        "created_utc":   created,
        "title":         raw.get("title",""),
        "score":         raw.get("score", 0),
        "upvote_ratio":  raw.get("upvote_ratio", 0),
        "num_comments":  raw.get("num_comments", 0),
        "subreddit":     raw.get("subreddit", subreddit),
        "author":        raw.get("author",""),
        "url":           "https://reddit.com" + raw.get("permalink",""),
        "img_url":       img,
        "img_local":     None,   # se rellena al descargar
        "flair":         raw.get("link_flair_text","") or "",
        "awards":        raw.get("total_awards_received", 0),
        "domain":        raw.get("domain",""),
        "selftext":      (raw.get("selftext") or "")[:500],
        "query_label":   query["label"],
        "query_id":      query["id"],
    }


# ─── Ventanas temporales ──────────────────────────────────────────────────────

def date_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())

def make_windows(start: date, end: date, window_days: int):
    """Genera ventanas de tiempo de más reciente a más antiguo."""
    windows = []
    cursor = end
    while cursor > start:
        ws = max(cursor - timedelta(days=window_days), start)
        windows.append((date_ts(ws), date_ts(cursor) + 86399, ws, cursor))
        cursor = ws
    return windows


# ─── Fetch Reddit nativo ──────────────────────────────────────────────────────

def fetch_reddit(session, query: dict, after_ts: int, before_ts: int,
                 delay_ms: int, subreddit: str) -> list[dict]:
    """
    Busca en un subreddit específico con restrict_sr=1.
    Filtra estrictamente por ventana de timestamps.
    """
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    posts = []
    after_cursor = None

    for page in range(10):
        if page > 0: sleep_j(delay_ms)
        params = {
            "q": query["q"],
            "sort": "top",          # top dentro de la ventana = más relevantes
            "t": "all",
            "limit": 100,
            "raw_json": 1,
            "restrict_sr": 1,       # SOLO en este subreddit
        }
        if after_cursor: params["after"] = after_cursor

        try:
            resp = session.get(url, params=params,
                               headers={"User-Agent": ua()}, timeout=15)
            if resp.status_code == 429:
                ra = int(resp.headers.get("Retry-After","60"))
                warn(f"  Rate limit 429 en r/{subreddit} — esperando {ra}s...")
                time.sleep(ra + random.uniform(3,8)); continue
            if resp.status_code in (403, 404):
                warn(f"  r/{subreddit}: HTTP {resp.status_code} — saltando")
                break
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"r/{subreddit}: {e}")

        children = data.get("data",{}).get("children",[])
        if not children: break

        stop = False
        for child in children:
            if child.get("kind") != "t3": continue
            p = child["data"]
            ts = p.get("created_utc", 0)
            if ts > before_ts: continue
            if ts < after_ts:  stop = True; break
            posts.append(p)

        after_cursor = data.get("data",{}).get("after")
        if not after_cursor or stop: break

    return posts


# ─── Arctic Shift (fuente primaria si disponible) ─────────────────────────────

ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api/posts/search"

def fetch_arctic(session, query: dict, after_ts: int, before_ts: int,
                 subreddit: str) -> list[dict]:
    params = {
        "q":        query["q"],
        "subreddit": subreddit,
        "after":    after_ts,
        "before":   before_ts,
        "limit":    100,
        "sort":     "score",
        "order":    "desc",
    }
    resp = session.get(ARCTIC_BASE, params=params,
                       headers={"User-Agent": ua()}, timeout=20)
    if resp.status_code == 429:
        ra = int(resp.headers.get("Retry-After","30"))
        warn(f"Arctic 429 — esperando {ra}s"); time.sleep(ra+3)
        resp = session.get(ARCTIC_BASE, params=params,
                           headers={"User-Agent": ua()}, timeout=20)
    resp.raise_for_status()
    return resp.json().get("data", [])

def probe_arctic(session) -> bool:
    try:
        r = session.get(ARCTIC_BASE, params={"q":"NVDA","limit":1},
                        headers={"User-Agent": ua()}, timeout=8)
        return r.status_code == 200
    except Exception: return False


# ─── Scraper principal ────────────────────────────────────────────────────────

class FinanceScraper:
    def __init__(self, start: date, end: date, queries: list[dict],
                 subreddits: list[str], top_n: int, window_days: int,
                 delay_ms: int, download_images: bool, output_dir: str,
                 source: str = "auto"):
        self.start    = start
        self.end      = end
        self.queries  = queries
        self.subs     = subreddits
        self.top_n    = top_n
        self.window   = window_days
        self.delay    = delay_ms
        self.dl_imgs  = download_images
        self.out_dir  = Path(output_dir)
        self.source   = source

        self.session   = build_session()
        self.seen      : set[str] = set()
        self.all_posts : list[dict] = []
        self.errors    = 0
        self.arctic_ok = False

        # Directorios de salida
        self.img_dir = self.out_dir / "imagenes"
        self.img_dir.mkdir(parents=True, exist_ok=True)

    # ── dedup + add ──────────────────────────────────────────────────────────
    def _add(self, raw: dict, query: dict, sub: str) -> Optional[dict]:
        pid = raw.get("id","")
        if not pid or pid in self.seen: return None
        self.seen.add(pid)
        p = make_post(raw, query, sub)
        self.all_posts.append(p)
        return p

    # ── fetch con fallback ───────────────────────────────────────────────────
    def _fetch(self, query, after_ts, before_ts, sub) -> list[dict]:
        if self.arctic_ok:
            try:
                return fetch_arctic(self.session, query, after_ts, before_ts, sub)
            except Exception as e:
                if self.source == "arctic": raise
                warn(f"  Arctic falló → Reddit nativo ({e})")
        return fetch_reddit(self.session, query, after_ts, before_ts, self.delay, sub)

    # ── descarga imagen ──────────────────────────────────────────────────────
    def _download(self, post: dict):
        if not post.get("img_url"): return
        ext = img_extension(post["img_url"])
        fname = f"{post['date']}_{post['id']}.{ext}"
        fpath = self.img_dir / fname
        if fpath.exists():
            post["img_local"] = str(fpath); return
        if download_image(post["img_url"], str(fpath), self.session):
            post["img_local"] = str(fpath)
            info(f"    📷 {fname}")
        else:
            warn(f"    No se pudo descargar imagen: {post['id']}")

    # ── run ──────────────────────────────────────────────────────────────────
    def run(self):
        hdr("NVIDIA Finance Reddit Scraper")
        print(col(f"  Período    : {self.start} → {self.end}", GR))
        print(col(f"  Subreddits : {len(self.subs)}", GR))
        for s in self.subs: print(col(f"               r/{s}", GR))
        print(col(f"  Queries    : {len(self.queries)}", GR))
        print(col(f"  Top/día    : {self.top_n} posts más relevantes", GR))
        print(col(f"  Ventana    : {self.window} día(s)", GR))
        print(col(f"  Imágenes   : {'Sí, en ./imagenes/' if self.dl_imgs else 'No'}", GR))

        windows = make_windows(self.start, self.end, self.window)
        total   = len(self.subs) * len(self.queries) * len(windows)

        print(col(f"\n  Requests estimados : ~{total:,}", GR))
        print(col(f"  Capacidad máxima   : ~{total*100:,} posts", G))

        # Detectar Arctic Shift
        if self.source in ("auto","arctic"):
            info("\nProbando Arctic Shift (archivo completo de Reddit)...")
            self.arctic_ok = probe_arctic(self.session)
            if self.arctic_ok: ok("Arctic Shift disponible ← usando como fuente principal")
            else: warn("Arctic Shift no disponible → Reddit nativo (sort=top)")

        done = 0
        for sub in self.subs:
            step(f"r/{sub}")
            for query in self.queries:
                info(f"Query: {query['label']}")
                q_added = 0
                for (after_ts, before_ts, ws, we) in windows:
                    done += 1
                    try:
                        raws = self._fetch(query, after_ts, before_ts, sub)
                    except Exception as e:
                        err(f"  Error: {e}"); self.errors += 1; raws = []

                    added = 0
                    for r in raws:
                        if self._add(r, query, sub): added += 1
                    q_added += added

                    if added:
                        info(f"  {ws}→{we}  +{added}  {pbar(done, total)}")
                    sleep_j(self.delay)

                if q_added:
                    ok(f"  {query['label']}: +{q_added} posts")
                sleep_j(int(self.delay * 1.2))

            sleep_j(int(self.delay * 1.8), f"Pausa tras r/{sub}")

        self._print_summary()

    # ── resumen + top por día ─────────────────────────────────────────────────
    def _print_summary(self):
        hdr("Resumen de scraping")
        ok(f"Posts únicos scrapeados : {len(self.all_posts)}")
        if self.errors: warn(f"Errores de red          : {self.errors}")

        # Distribución por subreddit
        sub_count: dict[str,int] = defaultdict(int)
        for p in self.all_posts: sub_count[p["subreddit"]] += 1
        section("Posts por subreddit")
        for s, n in sorted(sub_count.items(), key=lambda x: -x[1]):
            bar = "█" * min(n // 2, 45)
            print(col(f"    r/{s:<24} {bar} {n}", GR))

        # Top global
        section(f"Top 10 posts por score (global)")
        top = sorted(self.all_posts, key=lambda x: x["score"], reverse=True)[:10]
        for p in top:
            print(col(f"    ▲{p['score']:>6}  {p['date']}  "
                      f"r/{p['subreddit']:<18} {p['title'][:48]}", GR))

    # ── top N por día ─────────────────────────────────────────────────────────
    def top_by_day(self) -> dict[str, list[dict]]:
        """
        Para cada día en el rango, devuelve los top_n posts
        ordenados por score descendente, sin duplicados.
        """
        by_day: dict[str, list[dict]] = defaultdict(list)
        for p in self.all_posts:
            by_day[p["date"]].append(p)

        result = {}
        total_days = (self.end - self.start).days + 1
        for i in range(total_days):
            d = (self.start + timedelta(days=i)).isoformat()
            day_posts = sorted(by_day.get(d, []),
                               key=lambda x: x["score"], reverse=True)
            # Dedup por título (mismo contenido, distinto subreddit)
            seen_titles: set[str] = set()
            unique = []
            for p in day_posts:
                t = p["title"].lower().strip()
                if t not in seen_titles:
                    seen_titles.add(t)
                    unique.append(p)
            result[d] = unique[:self.top_n]
        return result

    # ── descargar imágenes de los top posts ───────────────────────────────────
    def download_top_images(self, top_by_day: dict):
        if not self.dl_imgs: return
        section("Descargando imágenes de posts top")
        total = sum(len(v) for v in top_by_day.values())
        done = 0
        for day, posts in sorted(top_by_day.items()):
            for p in posts:
                if p.get("img_url"):
                    self._download(p)
                done += 1
        imgs = sum(1 for p in self.all_posts if p.get("img_local"))
        ok(f"Imágenes descargadas: {imgs}  → {self.img_dir}/")

    # ── guardar resultados ────────────────────────────────────────────────────
    def save(self, top_by_day: dict):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.out_dir / f"nvidia_finance_{ts}"

        hdr("Exportando resultados")

        # ── JSON completo (todos los posts) ───────────────────────────────────
        all_sorted = sorted(self.all_posts,
                            key=lambda x: x["created_utc"], reverse=True)
        full_json = {
            "scraped_at":  datetime.now(timezone.utc).isoformat(),
            "date_range":  {"from": str(self.start), "to": str(self.end)},
            "subreddits":  self.subs,
            "total_posts": len(self.all_posts),
            "errors":      self.errors,
            "posts":       all_sorted,
        }
        jp = str(base) + "_todos.json"
        with open(jp,"w",encoding="utf-8") as f:
            json.dump(full_json, f, ensure_ascii=False, indent=2)
        ok(f"JSON completo   → {jp}")

        # ── JSON top por día ──────────────────────────────────────────────────
        top_json = {
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "date_range": {"from": str(self.start), "to": str(self.end)},
            "top_n":      self.top_n,
            "days":       {
                day: posts
                for day, posts in sorted(top_by_day.items())
                if posts
            }
        }
        tjp = str(base) + f"_top{self.top_n}_por_dia.json"
        with open(tjp,"w",encoding="utf-8") as f:
            json.dump(top_json, f, ensure_ascii=False, indent=2)
        ok(f"JSON top/día    → {tjp}")

        # ── CSV top por día (plano, ordenado) ────────────────────────────────
        flat_top = []
        for day in sorted(top_by_day):
            for rank, p in enumerate(top_by_day[day], 1):
                flat_top.append({**p, "day_rank": rank})

        fields = ["date", "day_rank", "id", "title", "score", "upvote_ratio",
                  "num_comments", "subreddit", "author", "url", "img_url",
                  "img_local", "flair", "awards", "query_label", "selftext"]
        csv_path = str(base) + f"_top{self.top_n}_por_dia.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(flat_top)
        ok(f"CSV top/día     → {csv_path}")

        # ── Reporte Markdown diario ───────────────────────────────────────────
        md_path = str(base) + "_reporte.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# NVIDIA Finance Reddit — Top {self.top_n} posts/día\n\n")
            f.write(f"**Período:** {self.start} → {self.end}  \n")
            f.write(f"**Subreddits:** {', '.join('r/'+s for s in self.subs)}  \n")
            f.write(f"**Scrapeado:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n\n")
            f.write("---\n\n")

            days_with_posts = {d: p for d, p in sorted(top_by_day.items()) if p}
            for day, posts in sorted(days_with_posts.items(), reverse=True):
                f.write(f"## {day}\n\n")
                for rank, p in enumerate(posts, 1):
                    img_md = ""
                    if p.get("img_local"):
                        img_md = f"\n  ![img]({p['img_local']})"
                    elif p.get("img_url"):
                        img_md = f"\n  ![img]({p['img_url']})"
                    f.write(
                        f"**#{rank}** · ▲{p['score']}  "
                        f"r/{p['subreddit']} · "
                        f"{p['num_comments']} comentarios  \n"
                        f"[{p['title']}]({p['url']}){img_md}  \n"
                        f"*Query: {p['query_label']}*\n\n"
                    )
                f.write("---\n\n")
        ok(f"Markdown diario → {md_path}")

        # ── Lista imágenes ─────────────────────────────────────────────────────
        img_urls = [p["img_url"] for p in self.all_posts if p.get("img_url")]
        if img_urls:
            il = str(base) + "_image_urls.txt"
            with open(il,"w",encoding="utf-8") as f: f.write("\n".join(img_urls))
            ok(f"URLs imágenes   → {il}  ({len(img_urls)})")
            if not self.dl_imgs:
                print(col(f"\n  Para descargar todas las imágenes:", CY))
                print(col(f"    wget -i {il} -P {self.img_dir}/", GR))

        imgs_dl = sum(1 for p in self.all_posts if p.get("img_local"))
        if imgs_dl:
            ok(f"Imágenes locales: {imgs_dl}  → {self.img_dir}/")

        print(col(f"\n  {'─'*50}", CY))
        print(col(f"  Directorio de salida: {self.out_dir.resolve()}", G+B))
        print(col(f"  {'─'*50}\n", CY))


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    today = date.today()
    p = argparse.ArgumentParser(
        description="NVIDIA Finance Reddit Scraper — top posts por día con imágenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subreddits financieros incluidos:
  wallstreetbets, stocks, investing, StockMarket, options,
  ValueInvesting, SecurityAnalysis, finance, algotrading,
  Daytrading, dividends, Economics, financialindependence

Queries NVIDIA (IDs 0-5):
  0: NVDA / NVIDIA General
  1: Earnings / Revenue / Guidance
  2: Blackwell / H100 / AI chips
  3: Jensen Huang / CEO
  4: Price targets / analyst ratings
  5: Short / puts / options NVDA

Ejemplos:
  python nvidia_finance_scraper.py                    # todo enero→hoy
  python nvidia_finance_scraper.py --top 10           # top 10 posts/día
  python nvidia_finance_scraper.py --top 3 --window 1 # ventanas de 1 día exacto
  python nvidia_finance_scraper.py --no-images        # sin descargar imágenes
  python nvidia_finance_scraper.py --from 2026-03-01  # desde marzo
  python nvidia_finance_scraper.py --query 0 1 5      # solo 3 queries
  python nvidia_finance_scraper.py --output-dir ./datos_nvda
        """
    )
    p.add_argument("--from",  dest="date_from", default="2026-01-01",
                   metavar="YYYY-MM-DD")
    p.add_argument("--to",    dest="date_to",   default=str(today),
                   metavar="YYYY-MM-DD")
    p.add_argument("--top",   type=int, default=5,
                   help="Posts más relevantes por día (default: 5)")
    p.add_argument("--window", type=int, default=3,
                   help="Ventana temporal en días (default: 3, usa 1 para máxima precisión)")
    p.add_argument("--delay",  type=int, default=1500,
                   help="Delay base entre requests en ms (default: 1500)")
    p.add_argument("--query",  nargs="+", type=int, metavar="ID",
                   help="IDs de queries (0-5). Default: todas")
    p.add_argument("--no-images", dest="no_images", action="store_true",
                   help="No descargar imágenes (solo guardar URLs)")
    p.add_argument("--source", choices=["auto","arctic","reddit"], default="auto")
    p.add_argument("--output-dir", default="./nvidia_finance_output",
                   metavar="DIR")
    p.add_argument("--no-color", action="store_true")
    return p.parse_args()


def main():
    global NO_COLOR
    args = parse_args()
    if args.no_color: NO_COLOR = True

    try:
        start = date.fromisoformat(args.date_from)
        end   = date.fromisoformat(args.date_to)
    except ValueError as e:
        err(f"Fecha inválida: {e}"); sys.exit(1)
    if start > end:
        err("--from debe ser anterior a --to"); sys.exit(1)

    queries = ([q for q in SEARCH_QUERIES if q["id"] in args.query]
               if args.query else SEARCH_QUERIES[:])
    if not queries:
        err("Ninguna query válida."); sys.exit(1)

    # Estimación
    windows_n = len(make_windows(start, end, args.window))
    total_req = len(FINANCE_SUBREDDITS) * len(queries) * windows_n
    est_min   = total_req * jitter(args.delay) / 60
    total_days = (end - start).days + 1

    hdr("Estimación")
    print(col(f"  Período       : {start} → {end} ({total_days} días)", GR))
    print(col(f"  Subreddits    : {len(FINANCE_SUBREDDITS)}", GR))
    print(col(f"  Queries       : {len(queries)}", GR))
    print(col(f"  Ventanas      : {windows_n} × {args.window} día(s)", GR))
    print(col(f"  Requests      : ~{total_req:,}", GR))
    print(col(f"  Tiempo est.   : ~{est_min:.0f}–{est_min*1.7:.0f} min", Y))
    print(col(f"  Top posts/día : {args.top}", G))
    print(col(f"  Imágenes      : {'Descarga automática → ./imagenes/' if not args.no_images else 'Solo URLs'}", G))

    confirm = input(col("\n  ¿Continuar? [s/N]: ", Y)).strip().lower()
    if confirm not in ("s","si","sí","y","yes"):
        print("Cancelado."); sys.exit(0)

    scraper = FinanceScraper(
        start=start, end=end,
        queries=queries,
        subreddits=FINANCE_SUBREDDITS,
        top_n=args.top,
        window_days=args.window,
        delay_ms=args.delay,
        download_images=not args.no_images,
        output_dir=args.output_dir,
        source=args.source,
    )

    try:
        scraper.run()
    except KeyboardInterrupt:
        warn("\n\nInterrumpido por el usuario. Guardando lo obtenido...")

    top = scraper.top_by_day()

    # Descargar imágenes solo de los posts top
    scraper.download_top_images(top)

    scraper.save(top)


if __name__ == "__main__":
    main()
