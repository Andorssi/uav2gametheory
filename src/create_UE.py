import random
from collections import defaultdict

U = 200
I = 49

def create_client_file(filename):
    with open(filename, "w") as f:
        for u in range(U):
            area = random.randrange(I)
            f.write(f"{u} {area}\n")

def count_client(filename):
    area_count = defaultdict(int)
    with open(filename, "r") as f:
        for line in f:
            u, area = map(int, line.split())
            area_count[area] += 1
    return area_count

def main():

    # create_client_file("client_area.txt")
    client_count = count_client("client_area.txt")

    for area in sorted(client_count):
        print(f"領域{area}: {client_count[area]}")

if __name__ == "__main__":
    main()