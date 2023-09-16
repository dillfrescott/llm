import pyarrow.parquet as pq

# Path to your Parquet file
parquet_file = "1M-GPT4-Augmented.parquet"

# Read the Parquet file
table = pq.read_table(parquet_file)

# Path to the output TXT file
txt_output_file = "output.txt"

# Convert the Parquet table to a Pandas DataFrame
df = table.to_pandas()

# Save the DataFrame to a TXT file
df.to_csv(txt_output_file, sep='\t', index=False)  # You can specify the separator as needed
