
#!/bin/bash

mkdir -p data/output/spliceai/
mkdir -p data/output/annovar/humandb/
mkdir -p data/annotation_database/

FA_GZ="data/output/spliceai/hg38.fa.gz"
FA_OUT="data/output/spliceai/hg38.fa"

if [ -f "$FA_OUT" ]; then
    echo "hg38.fa sudah diekstrak."
else
    if [ ! -f "$FA_GZ" ]; then
        wget -O "$FA_GZ" https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
    fi
    gunzip -f "$FA_GZ"
fi

CHROM_BEDG="data/annotation_database/master38.chromhmm.bedg"
CHROM_TAR="data/annotation_database/master38.chromhmm.bedg.tar.gz"

if [ -f "$CHROM_BEDG" ]; then
    echo "ChromHMM sudah diekstrak."
else
    if [ -f "$CHROM_TAR" ]; then
        tar -zxvf "$CHROM_TAR" -C data/annotation_database/
    fi
fi

BASE_URL="http://www.openbioinformatics.org/annovar/download"
ANNOVAR_DB="data/output/annovar/humandb"

download_and_extract() {
    local file_gz="$ANNOVAR_DB/$1"
    local file_out="$ANNOVAR_DB/$2"
    local url="$BASE_URL/$1"

    if [ -f "$file_out" ]; then
        echo "Berkas $2 sudah siap."
    else
        if [ ! -f "$file_gz" ]; then
            wget -O "$file_gz" "$url"
        fi
        gunzip -f -k "$file_gz"
    fi
}

if [ -f "data/output/annovar/humandb/hg38_phastConsElements100way.txt" ]; then
    echo "hg38_phastConsElements100way.txt sudah siap."
else
    wget -O data/output/annovar/humandb/hg38_phastConsElements100way.txt.gz http://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/phastConsElements100way.txt.gz
    gunzip -f -k data/output/annovar/humandb/hg38_phastConsElements100way.txt.gz
fi

download_and_extract "hg38_refGene.txt.gz" "hg38_refGene.txt"
download_and_extract "hg38_refGeneMrna.fa.gz" "hg38_refGeneMrna.fa"
download_and_extract "hg38_refGeneVersion.txt.gz" "hg38_refGeneVersion.txt"
download_and_extract "hg38_dbnsfp33a.txt.gz" "hg38_dbnsfp33a.txt"
download_and_extract "hg38_dbnsfp33a.txt.idx.gz" "hg38_dbnsfp33a.txt.idx"
download_and_extract "hg38_dbnsfp42a.txt.gz" "hg38_dbnsfp42a.txt"
download_and_extract "hg38_dbnsfp42a.txt.idx.gz" "hg38_dbnsfp42a.txt.idx"
download_and_extract "hg38_gnomad30_genome.txt.gz" "hg38_gnomad30_genome.txt"
download_and_extract "hg38_gnomad30_genome.txt.idx.gz" "hg38_gnomad30_genome.txt.idx"

GENCC_DIR="data/annotation_database"
mkdir -p "$GENCC_DIR"

HEADERS_FILE="$GENCC_DIR/gencc_headers.txt"
OUTPUT_FILE="$GENCC_DIR/gencc_submissions.csv"
URL="https://thegencc.org/download/action/submissions-export-csv?format=new"

if [ -f "$HEADERS_FILE" ] && [ -f "$OUTPUT_FILE" ]; then
  ETAG=$(grep -i '^ETag:' "$HEADERS_FILE" | sed 's/ETag: //i' | tr -d '"\r\n')
  HTTP_CODE=$(curl -s -D "$HEADERS_FILE" -H "If-None-Match: \"$ETAG\"" --write-out "%{http_code}" -o "$OUTPUT_FILE" "$URL")
  
  if [ "$HTTP_CODE" = "304" ]; then
    echo "Database GenCC sudah versi terbaru."
  elif [ "$HTTP_CODE" = "200" ]; then
    echo "Sukses memperbarui database GenCC."
  else
    echo "Menggunakan data lokal yang ada."
  fi
else
  curl -s -D "$HEADERS_FILE" -o "$OUTPUT_FILE" "$URL"
  echo "Database GenCC sukses disimpan."
fi
