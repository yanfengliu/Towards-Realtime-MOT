import subprocess

for i in range(10):
    input_str =  f'C:/Users/yliu60/Documents/GitHub/embedding_tracking/dataset2/6_shapes/test/seq_{i}/video/video.avi'
    output_str = f'temp/{i}'
    subprocess.run(['python', 'demo.py', '--input-video', input_str, '--weights', 
        'C:/Users/yliu60/Documents/GitHub/Towards-Realtime-MOT/weights/epoch-10.pt',
        '--output-format', 'video', '--output-root', output_str])