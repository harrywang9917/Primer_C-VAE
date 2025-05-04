#!/bin/bash

input_file="/Users/awc789/Downloads/TEST/output_accessions.txt"
output_dir="/Users/awc789/Downloads/TEST/downloaded_sequences"

# 检查 datasets 命令是否可用
if ! command -v datasets &> /dev/null
then
    echo "Error: datasets command not found. Please install NCBI datasets command-line tool."
    echo "You can install it using one of the following methods:"
    echo "1. conda install -c conda-forge ncbi-datasets-cli"
    echo "2. brew install datasets"
    exit 1
fi

# 确保输出目录存在
mkdir -p "$output_dir"

while IFS= read -r accession
do
    echo "Downloading assembly for $accession"
    
    # 使用 datasets 命令下载
    if ! datasets download genome accession $accession \
        --include genome \
        --filename ${output_dir}/${accession}.zip
    then
        echo "Error: Failed to download $accession. Skipping to next assembly."
        continue
    fi
    
    # 解压下载的文件
    if ! unzip -o ${output_dir}/${accession}.zip -d ${output_dir}/${accession}_temp
    then
        echo "Error: Failed to unzip $accession. Skipping to next assembly."
        rm -f ${output_dir}/${accession}.zip
        continue
    fi
    
    # 查找并移动 .fna 文件，删除其他所有文件
    find ${output_dir}/${accession}_temp -name "*.fna" -exec mv {} ${output_dir}/${accession}_genomic.fna \;
    
    # 如果没有找到 .fna 文件，打印警告
    if [ ! -f ${output_dir}/${accession}_genomic.fna ]; then
        echo "Warning: No .fna file found for $accession."
    fi
    
    # 清理临时文件和其他下载的文件
    rm -rf ${output_dir}/${accession}_temp
    rm -f ${output_dir}/${accession}.zip
    rm -f ${output_dir}/${accession}_genomic.gff
    rm -f ${output_dir}/${accession}_protein.faa
    
    echo "Finished processing $accession"
    
    # 添加一个短暂的延迟，以避免过于频繁的请求
    sleep 1
done < "$input_file"

echo "All assemblies have been downloaded and processed. Only .fna files have been retained."