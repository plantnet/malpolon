awk '
    NR==FNR { words[$1]; next } 
    {
        keep = 0
        for (i = 114; i <= 118; i++) {
            if ($i in words) {
                keep = 1
                break
            }
        }
        if (keep) print
    }
' file2.txt file1.txt > filtered_file.txt
# file
