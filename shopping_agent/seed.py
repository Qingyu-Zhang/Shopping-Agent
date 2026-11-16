from .core import timestamp

# Spec fixtures are attributed to the fictional manufacturer's catalog.
ITEMS = (
    ("air-16", "air", "Aurora Air 14", "Aurora", "laptop", "16 GB Silver", 71900,
     {"weight_g": 1290, "ram_gb": 16, "video": {"dp_alt": 14}},
     "Portable lightweight business laptop with USB-C DisplayPort Alt Mode."),
    ("air-8", "air", "Aurora Air 14", "Aurora", "laptop", "8 GB Blue", 61900,
     {"weight_g": 1290, "ram_gb": 8, "video": {}},
     "Commuter laptop with a USB-C data port."),
    ("cedar-16", "cedar", "Cedar Book 13", "Cedar", "laptop", "16 GB Gray", 80900,
     {"weight_g": 1080, "ram_gb": 16, "video": {"dp_alt": 14, "thunderbolt": 4}},
     "Ultralight laptop for travel and business with Thunderbolt 4."),
    ("summit-32", "summit", "Summit Studio 15", "Summit", "laptop", "32 GB Black", 105900,
     {"weight_g": 1840, "ram_gb": 32, "video": {"dp_alt": 14}},
     "High-performance mobile workstation."),
    ("link-mini", "mini", "Link Mini Hub", "Link Works", "dock", "6-port Gray", 4900,
     {"weight_g": 82, "inputs": {"dp_alt": {"min": 14, "outputs": {"hdmi": 60}}}},
     "Portable USB-C dock HDMI 4K@60Hz with a DP 1.4 host."),
    ("link-pro", "pro", "Link Pro Dock", "Link Works", "dock", "9-port Silver", 8900,
     {"weight_g": 137, "inputs": {"dp_alt": {"min": 14, "outputs": {"hdmi": 60, "dp": 60}}}},
     "Desktop dock with HDMI and DisplayPort 4K 60 Hz."),
    ("link-lite", "lite", "Link Lite Hub", "Link Works", "dock", "4-port Gray", 2900,
     {"weight_g": 50, "inputs": {"dp_alt": {"min": 12, "outputs": {"hdmi": 30}}}},
     "Affordable HDMI 4K@30Hz travel adapter."),
    ("link-nano", "nano", "Link Nano Dock", "Link Works", "dock", "4-port Titanium", 6900,
     {"weight_g": 58, "inputs": {"dp_alt": {"min": 14, "outputs": {"hdmi": 60}}}},
     "Very light travel USB-C dock for 4K 60Hz HDMI monitors."),
    ("vista-27", "vista", "Vista 27 Monitor", "Vista", "monitor", "Black", 24900,
     {"weight_g": 5260, "monitor_ports": ["hdmi", "dp"], "refresh_rate_hz": 60}, "External 4K 60Hz monitor."),
    ("fresh300", "fresh", "Fresh 300 Air Filter", "Fresh", "filter", "One pack", 1599,
     {"compatible_with": ["Fresh 300"]}, "Replacement air purifier filter for periodic replenishment."),
    ("tuna-cat", "tuna", "Ocean Tuna Cat Food", "Ocean", "pet_food", "1.5 kg", 1999,
     {"weight_g": 1500}, "Tuna dry cat food for monthly restocking."),
)

USERS = {
    "alex": {"id": "alex", "name": "Alex", "addresses": {"home": "Seattle, WA", "office": "Portland, OR"},
             "payment_methods": {"saved-card": "Demo card ending in 4242"}},
    "sam": {"id": "sam", "name": "Sam", "addresses": {"home": "Boston, MA"},
            "payment_methods": {"saved-card": "Demo card ending in 4242"}},
}


def seed_catalog(store):
    with store.transaction(True) as tx:
        for sku, product, name, brand, cat, variant, amount, specs, description in ITEMS:
            tx.put("products", product, {
                "id": product, "product_id": product, "source": "Demo manufacturer catalog",
                "title": name, "brand": brand, "category": cat, "description": description,
            }, once=True)
            tx.put("catalog", sku, {
                "id": sku, "sku_id": sku, "product_id": product, "title": name, "brand": brand,
                "category": cat, "variant": variant, "snapshot_price": amount, "specs": specs,
                "source": "Demo manufacturer catalog", "source_version": "1",
                "description": description, "market": "US", "active": True,
            }, owner=product, once=True)
        for sku, review in (
            ("air-16", "Light enough to carry to work daily. The keyboard is comfortable."),
            ("cedar-16", "Very easy to travel with. The screen is bright."),
            ("link-mini", "Fits into my laptop bag. The included cable is short."),
            ("link-nano", "Very compact. It gets warm in long meetings."),
            ("link-pro", "Plenty of ports but heavy for travel."),
        ):
            item = tx.get("catalog", sku)
            tx.put("evidence", "review-" + sku, {
                "id": "review-" + sku, "sku_id": sku, "product_id": item["product_id"],
                "kind": "review", "source": "Demo product review", "text": review, "rating": 4,
            }, owner=item["product_id"], once=True)


def seed_platform(store):
    with store.transaction(True) as tx:
        for user, info in USERS.items():
            tx.put("accounts", user, info, once=True)
        for index, (sku, _, name, brand, _, variant, amount, _, _) in enumerate(ITEMS):
            offer = {
                "id": sku + "-official", "sku_id": sku, "title": name, "variant": variant,
                "seller": brand, "unit_price": amount, "stock": 12, "currency": "USD",
                "shipping": 0, "lead_hours": 12 + index * 2, "updated_at": timestamp(),
            }
            if tx.put("offers", offer["id"], offer, owner=sku, once=True):
                key = f'{offer["id"]}-initial'
                tx.put("price_history", key, {"id": key, **offer, "at": offer["updated_at"]},
                       owner=offer["id"], once=True)
        offer = {**tx.require("offers", "air-16-official"),
                 "id": "air-16-partner", "seller": "Tech Market", "unit_price": 69900, "lead_hours": 96}
        if tx.put("offers", offer["id"], offer, owner=offer["sku_id"], once=True):
            key = f'{offer["id"]}-initial'
            tx.put("price_history", key, {"id": key, **offer, "at": offer["updated_at"]},
                   owner=offer["id"], once=True)
