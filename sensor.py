import random
import time

def get_vehicle_count():
    """Simulate vehicle count from LiDAR."""
    return random.randint(0, 20)

if __name__ == "__main__":
    while True:
        print("Vehicle count:", get_vehicle_count())
        time.sleep(2)
