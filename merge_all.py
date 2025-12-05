import pickle

# def merge_jam_data(
#     files=[
#         "data/jam_train_data_1.pkl",
#         "data/jam_train_data_2.pkl",
#         "data/jam_train_data_3.pkl",
#         "data/jam_train_data_4.pkl",
#         "data/jam_train_data_5.pkl",
#     ],
#     output_file="data/jam_train_data_all_zigzag_70.pkl"
# ):

def merge_jam_data(
    files=[
        "data/jam_train_data_all_zigzag.pkl",
        "data/jam_train_data_6.pkl",
        "data/jam_train_data_7.pkl",
    ],
    output_file="data/jam_train_data_all_zigzag_70.pkl"
):
    print("Merging files:")
    for f in files:
        print("  ", f)

    merged = []

    for f in files:
        with open(f, "rb") as fp:
            data = pickle.load(fp)
            print(f"Loaded {f} with {len(data)} episodes")
            merged.extend(data)

    with open(output_file, "wb") as fp:
        pickle.dump(merged, fp)

    print("\nMerged total:", len(merged), "episodes")
    print("Saved to:", output_file)


if __name__ == "__main__":
    merge_jam_data()
