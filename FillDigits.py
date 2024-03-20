import csv

# Function to format numbers to have three digits
def format_number(number):
    return str(number).zfill(3)

# Read the original CSV file
with open('test.csv', 'r') as input_file:
    reader = csv.reader(input_file)
    rows = list(reader)

# Process each row and modify numbers
for row in rows:
    for i, item in enumerate(row):
        try:
            # Try to convert the item to an integer
            number = int(item)
            # Format the number to have three digits
            row[i] = format_number(number)
        except ValueError:
            # If the item is not a number, leave it unchanged
            pass

        # Remove carriage returns from the item
        row[i] = row[i].strip("\r")

# Write the modified data to a new CSV file
with open('modified.csv', 'w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerows(rows)