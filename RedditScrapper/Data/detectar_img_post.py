import html


def normalize_url(url):
    if not url:
        return None
    if not isinstance(url, str):
        url = str(url)
    return html.unescape(url.strip())


def is_probably_image_url(url):
    if not url:
        return False

    url = url.lower()

    # extensiones típicas
    if any(ext in url for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]):
        return True

    # hosts típicos de imagen de Reddit/Imgur
    if any(host in url for host in ["i.redd.it", "preview.redd.it", "i.imgur.com"]):
        return True

    return False


def extract_image_urls(post):
    urls = []

    # 1. preview.images[].source.url  ← muy importante
    preview = post.get("preview")
    if isinstance(preview, dict):
        images = preview.get("images", [])
        if isinstance(images, list):
            for img in images:
                if not isinstance(img, dict):
                    continue

                source = img.get("source", {})
                if isinstance(source, dict):
                    url = normalize_url(source.get("url"))
                    if url:
                        urls.append(url)

                resolutions = img.get("resolutions", [])
                if isinstance(resolutions, list):
                    for r in resolutions:
                        if isinstance(r, dict):
                            url = normalize_url(r.get("url"))
                            if url:
                                urls.append(url)

    # 2. url_overridden_by_dest
    url = normalize_url(post.get("url_overridden_by_dest"))
    if is_probably_image_url(url):
        urls.append(url)

    # 3. url
    url = normalize_url(post.get("url"))
    if is_probably_image_url(url):
        urls.append(url)

    # 4. galería Reddit
    media_metadata = post.get("media_metadata")
    if isinstance(media_metadata, dict):
        for _, item in media_metadata.items():
            if not isinstance(item, dict):
                continue

            # versión source
            s = item.get("s", {})
            if isinstance(s, dict):
                url = normalize_url(s.get("u"))
                if url:
                    urls.append(url)

            # versiones p (preview list)
            p_list = item.get("p", [])
            if isinstance(p_list, list):
                for p in p_list:
                    if isinstance(p, dict):
                        url = normalize_url(p.get("u"))
                        if url:
                            urls.append(url)

    # 5. fallback: si dice que es imagen y hay preview, aunque la url no tenga extensión
    if post.get("post_hint") == "image":
        preview = post.get("preview")
        if isinstance(preview, dict):
            images = preview.get("images", [])
            for img in images:
                if isinstance(img, dict):
                    source = img.get("source", {})
                    if isinstance(source, dict):
                        url = normalize_url(source.get("url"))
                        if url:
                            urls.append(url)

    # quitar duplicados manteniendo orden
    clean_urls = []
    seen = set()
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            clean_urls.append(u)

    return clean_urls