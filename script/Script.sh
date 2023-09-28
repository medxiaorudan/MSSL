#!/bin/bash

# Define a function to run training files
run_train_files() {
    for file in "$@"; do
        case $file in
            "main_SSL_multi") python main_SSL_multi.py ;;
            "main_SSL_single") python main_SSL_single.py ;;
            "main_classify_single") python main_classify_single.py ;;
            "main_supervise_multi") python main_supervise_multi.py ;;
            "main_supervise_seg_single") python main_supervise_seg_single.py ;;
            *) echo "Unknown training file: $file" ;;
        esac
    done
}

# Define a function to run testing files
run_test_files() {
    for file in "$@"; do
        case $file in
            "test_SSL_multi") python test_SSL_multi.py ;;
            "test_SSL_single") python test_SSL_single.py ;;
            "test_classify_single") python test_classify_single.py ;;
            "test_supervise_multi") python test_supervise_multi.py ;;
            "test_supervise_seg_single") python test_supervise_seg_single.py ;;
            *) echo "Unknown testing file: $file" ;;
        esac
    done
}

# Main function
main() {
    if [ "$1" == "train" ]; then
        shift 1
        run_train_files "$@"
    elif [ "$1" == "test" ]; then
        shift 1
        run_test_files "$@"
    else
        echo "The first argument should be 'train' or 'test'"
    fi
}

main "$@"
