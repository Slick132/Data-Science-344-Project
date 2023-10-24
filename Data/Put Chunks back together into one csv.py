import glob

def combine_csv(input_dir, output_file):
    """
    Combines multiple CSV files from a directory into a single CSV file.

    Parameters:
    - input_dir (str): The directory containing the smaller CSV files.
    - output_file (str): The path where the combined CSV file will be saved.

    Returns:
    - None
    """

    # Get a list of all the CSV files to combine
    csv_files = sorted(glob.glob(os.path.join(input_dir, 'part_*.csv')))

    # Open the output file for writing
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)

        # Process the first file, including the header
        with open(csv_files[0], 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            for row in reader:
                writer.writerow(row)

        # Process the remaining files, skipping their headers
        for file in csv_files[1:]:
            with open(file, 'r', newline='', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                next(reader)  # skip header
                for row in reader:
                    writer.writerow(row)

# Directory and file paths
input_dir = r'C:\Users\Chris\OneDrive - Stellenbosch University\Documents\University\Third Year\Semester 2\Data Science 344\Project Github Repo\Data-Science-344-Project\Data\Data subsets'
output_file = r'C:\Users\Chris\OneDrive - Stellenbosch University\Documents\University\Third Year\Semester 2\Data Science 344\Project Github Repo\Data-Science-344-Project\Data\Data subsets\CombinedData.csv'

# Combining the CSV files
combine_csv(input_dir, output_file)
