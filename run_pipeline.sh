#!/bin/bash

# Generic Data Ingestion Pipeline Runner
# Helper script to run the pipeline with common configurations

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_usage() {
    echo "Usage: $0 [PRESET|CUSTOM]"
    echo ""
    echo "PRESETS:"
    echo "  exchange-rates    - Fetch exchange rates from CMF Chile API"
    echo "  sql-import        - Import data from SQL database"
    echo "  gcs-import        - Import data from Google Cloud Storage"
    echo "  local-import      - Import data from local file"
    echo ""
    echo "CUSTOM:"
    echo "  custom            - Run with custom parameters (interactive)"
    echo ""
    echo "Examples:"
    echo "  $0 exchange-rates"
    echo "  $0 sql-import"
    echo "  $0 custom"
}

run_exchange_rates() {
    echo -e "${GREEN}Running Exchange Rates Pipeline${NC}"
    echo ""

    read -p "Mode (incremental/full/backfill) [incremental]: " mode
    mode=${mode:-incremental}

    if [ "$mode" = "backfill" ]; then
        read -p "Backfill days [7]: " days
        days=${days:-7}

        python pipelines/data_ingestion_pipeline.py \
            --source api \
            --mode backfill \
            --backfill-days $days
    else
        python pipelines/data_ingestion_pipeline.py \
            --source api \
            --mode $mode
    fi
}

run_sql_import() {
    echo -e "${GREEN}Running SQL Import Pipeline${NC}"
    echo ""

    read -p "SQL Query: " query
    if [ -z "$query" ]; then
        echo -e "${RED}Query cannot be empty${NC}"
        exit 1
    fi

    read -p "Environment (dev/prod) [dev]: " env
    env=${env:-dev}

    read -p "Mode (incremental/full) [full]: " mode
    mode=${mode:-full}

    python pipelines/data_ingestion_pipeline.py \
        --source sql \
        --query "$query" \
        --environment $env \
        --mode $mode
}

run_gcs_import() {
    echo -e "${GREEN}Running GCS Import Pipeline${NC}"
    echo ""

    read -p "GCS blob path (bucket/path/file.csv): " blob_path
    if [ -z "$blob_path" ]; then
        echo -e "${RED}Blob path cannot be empty${NC}"
        exit 1
    fi

    read -p "File format (csv/parquet/json) [csv]: " format
    format=${format:-csv}

    python pipelines/data_ingestion_pipeline.py \
        --source gcs \
        --blob-path "$blob_path" \
        --file-format $format \
        --mode full
}

run_local_import() {
    echo -e "${GREEN}Running Local Import Pipeline${NC}"
    echo ""

    read -p "Local file path: " file_path
    if [ -z "$file_path" ]; then
        echo -e "${RED}File path cannot be empty${NC}"
        exit 1
    fi

    if [ ! -f "$file_path" ]; then
        echo -e "${RED}File not found: $file_path${NC}"
        exit 1
    fi

    read -p "File format (csv/parquet/json/excel) [csv]: " format
    format=${format:-csv}

    python pipelines/data_ingestion_pipeline.py \
        --source local \
        --file-path "$file_path" \
        --file-format $format \
        --mode full
}

run_custom() {
    echo -e "${GREEN}Running Custom Pipeline${NC}"
    echo ""

    echo "Select source type:"
    echo "1) API"
    echo "2) SQL"
    echo "3) GCS"
    echo "4) Local"
    read -p "Choice [1]: " choice
    choice=${choice:-1}

    case $choice in
        1) source="api" ;;
        2) source="sql" ;;
        3) source="gcs" ;;
        4) source="local" ;;
        *) echo "Invalid choice"; exit 1 ;;
    esac

    read -p "Mode (incremental/full/backfill) [incremental]: " mode
    mode=${mode:-incremental}

    read -p "Config file [config/config.yaml]: " config
    config=${config:-config/config.yaml}

    read -p "Secrets file [config/secrets.yaml]: " secrets
    secrets=${secrets:-config/secrets.yaml}

    # Build command
    cmd="python pipelines/data_ingestion_pipeline.py --source $source --mode $mode --config $config --secrets $secrets"

    # Source-specific params
    if [ "$source" = "sql" ]; then
        read -p "SQL Query: " query
        read -p "Environment (dev/prod): " env
        cmd="$cmd --query \"$query\" --environment $env"
    elif [ "$source" = "gcs" ]; then
        read -p "GCS blob path: " blob_path
        read -p "File format: " format
        cmd="$cmd --blob-path \"$blob_path\" --file-format $format"
    elif [ "$source" = "local" ]; then
        read -p "File path: " file_path
        read -p "File format: " format
        cmd="$cmd --file-path \"$file_path\" --file-format $format"
    fi

    if [ "$mode" = "backfill" ]; then
        read -p "Backfill days: " days
        cmd="$cmd --backfill-days $days"
    fi

    echo ""
    echo -e "${YELLOW}Running: $cmd${NC}"
    echo ""
    eval $cmd
}

# Main
if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

case $1 in
    exchange-rates)
        run_exchange_rates
        ;;
    sql-import)
        run_sql_import
        ;;
    gcs-import)
        run_gcs_import
        ;;
    local-import)
        run_local_import
        ;;
    custom)
        run_custom
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown preset: $1${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Pipeline completed successfully!${NC}"
