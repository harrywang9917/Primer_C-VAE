import json


def process_jsonl(input_file, output_file):
    accessions = []

    with open(input_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                accession = data.get('accession')
                if accession:
                    accessions.append(accession)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from line: {line}")

    with open(output_file, 'w') as f:
        for accession in accessions:
            f.write(f"{accession}\n")

    print(f"Processed {len(accessions)} accessions and saved to {output_file}")


if __name__ == "__main__":
    input_file = "/Users/awc789/Downloads/TEST/assembly_data_report.jsonl"  # 替换为你的输入文件路径
    output_file = "/Users/awc789/Downloads/TEST/output_accessions.txt"  # 替换为你想要的输出文件路径

    process_jsonl(input_file, output_file)