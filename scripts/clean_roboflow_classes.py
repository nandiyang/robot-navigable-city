"""
clean_roboflow_classes.py
=========================
Deletes all wrongly-generated timestamp classes from your Roboflow project.
Keeps only: obstacle, surface_defect (and any other real classes you define)

Setup:
  pip install roboflow requests

Usage:
  1. Fill in your API key and project details below
  2. Run: python clean_roboflow_classes.py
"""

import requests

# ──────────────────────────────────────────────
# ✏️  FILL THESE IN
# ──────────────────────────────────────────────
API_KEY      = "sH5tPvY5bVlL6GnnuzdB"   # from app.roboflow.com → profile → Roboflow API
WORKSPACE    = "nandis-workspace"            # visible in your Roboflow URL
PROJECT_NAME = "robot-navigable-city"        # your project name

# Classes to KEEP — everything else gets deleted
KEEP_CLASSES = {"obstacle", "surface_defect"}
# ──────────────────────────────────────────────


def get_classes(workspace, project, api_key):
    """Fetch all classes via project info endpoint."""
    url = f"https://api.roboflow.com/{workspace}/{project}"
    resp = requests.get(url, params={"api_key": api_key})
    resp.raise_for_status()
    data = resp.json()
    # Classes are under project > classes
    classes = data.get("project", {}).get("classes", [])
    return classes  # list of class name strings


def delete_class(workspace, project, api_key, class_name):
    """Delete a single class and all its annotations."""
    url = f"https://api.roboflow.com/{workspace}/{project}/classes/{requests.utils.quote(class_name, safe='')}"
    resp = requests.delete(url, params={"api_key": api_key})
    return resp.status_code


def main():
    # Confirm once upfront before looping
    confirm = input(f"This will delete all timestamp classes from {WORKSPACE}/{PROJECT_NAME}, keeping only {KEEP_CLASSES}. Continue? (yes/no): ")
    if confirm.strip().lower() != "yes":
        print("Cancelled.")
        return

    total_deleted = 0
    round_num     = 0

    # Loop until no more timestamp classes remain
    # (API returns classes in pages so we keep fetching until clean)
    while True:
        round_num += 1
        print(f"\nRound {round_num}: fetching classes...")
        all_classes = get_classes(WORKSPACE, PROJECT_NAME, API_KEY)
        print(f"  Found {len(all_classes)} classes")

        to_delete = [c for c in all_classes if c not in KEEP_CLASSES]

        if not to_delete:
            print("  No more classes to delete — all clean!")
            break

        print(f"  Deleting {len(to_delete)} classes...")
        for i, class_name in enumerate(to_delete, 1):
            status = delete_class(WORKSPACE, PROJECT_NAME, API_KEY, class_name)
            if status in [200, 204]:
                total_deleted += 1
                print(f"  [{i}/{len(to_delete)}] Deleted: {class_name}")
            else:
                print(f"  [{i}/{len(to_delete)}] FAILED ({status}): {class_name}")

    print(f"\nDone — {total_deleted} classes deleted total")
    remaining = get_classes(WORKSPACE, PROJECT_NAME, API_KEY)
    print(f"Remaining classes: {remaining}")


if __name__ == "__main__":
    main()
