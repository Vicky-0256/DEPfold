import os

def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        content1 = f1.readlines()
        content2 = f2.readlines()
        
        if content1 == content2:
            return True
        else:
            for line_num, (line1, line2) in enumerate(zip(content1, content2), 1):
                if line1 != line2:
                    a=1
                    #print(f"Line {line_num} differs between {file1} and {file2}:")
                    # print(f"{file1}: {line1}")
                    # print(f"{file2}: {line2}")
            return False

def main(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    prefix_set = set()

    for file in files:
        if file.endswith("_p1.ct"):
            prefix = file.rsplit("_p1.ct", 1)[0]
            prefix_set.add(prefix)
    
    for prefix in prefix_set:
        file1 = os.path.join(directory, f"{prefix}_p1.ct")
        file10 = os.path.join(directory, f"{prefix}_p10.ct")

        if os.path.exists(file1) and os.path.exists(file10):
            # print(f"Comparing {file1} with {file10}...")
            if compare_files(file1, file10):
                print(f"Files {file1} and {file10} are identical.")
            else:
                # print(f"Files {file1} and {file10} are different.")
                a=1
            print("")

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    main(directory)
