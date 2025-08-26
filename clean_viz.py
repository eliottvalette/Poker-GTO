import os

viz_folders = [f for f in os.listdir("viz") if "viz_iter" in f]

for folder in viz_folders:
    for file in os.listdir(f"viz/{folder}"):
        os.remove(f"viz/{folder}/{file}")
    os.rmdir(f"viz/{folder}")

ranges_temp_jsons = [f for f in os.listdir("ranges") if "ranges_" in f]

for file in ranges_temp_jsons:
    os.remove(f"ranges/{file}")

policy_temp_jsons = [f for f in os.listdir("policy") if "avg_policy_iter" in f]

for file in policy_temp_jsons:
    os.remove(f"policy/{file}")