import csv

def extract_ids(input_csv, output_txt):
    ids = []
    
    try:
        with open(input_csv, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            
            for row in reader:
                if row:  
                    ids.append(row[0])
        
        with open(output_txt, 'w', encoding='utf-8') as txtfile:
            txtfile.write('\n'.join(ids))
            
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    input_csv = "./val/Electronic.csv"  
    output_txt = "./val/Electronic_split_sequence_names.txt" 
    
    extract_ids(input_csv, output_txt)