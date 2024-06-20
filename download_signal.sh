exp_name="$1"

yt-dlp -a ${exp_name}_signal.txt -o "input_${exp_name}/%(id)s.%(ext)s" -f bestaudio