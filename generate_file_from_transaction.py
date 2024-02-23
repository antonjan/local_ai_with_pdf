'''
This program will read an input.csv file and will create sentence from it and create and output.csv file with it
'''
import csv
def display_usage_info():
    print("Will be reading input.csv file and generating an output.csv that now have been sentence\n")    
def process_csv(file_path,output_file_path):
    out_file_path = open(output_file_path, "w")
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            sentence = "{Name} {Surname} is a {Title} in {Industry} from {Location} and he is paying a {Transaction type} transaction and is paying an amount of {Amount} in currency {Currency} to {Beneficiary Name} {Beneficiary Surname} to account number {Account Number} in a currency of {Beneficiary Currency} and the beneficiary wants value on this date {Value Date},\n".format(**row)
            print(sentence)
            out_file_path.write(sentence) 
    out_file_path.close()
if __name__ == "__main__":
    display_usage_info()
    file_path = "input.csv" 
    output_file_path = "output.csv"
    process_csv(file_path,output_file_path)
    
