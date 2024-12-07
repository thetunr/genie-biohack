import os
import pandas as pd
from PIL import Image
import subprocess

csv_file_path = '/teamspace/studios/this_studio/fitzpatrick17k.csv'
output_folder = '/teamspace/studios/this_studio/hash_image_2'

df = pd.read_csv(csv_file_path)
df = df.dropna(subset=['url'])

# df = df[['md5hash', 'fitzpatrick_scale', 'fitzpatrick_centaur', 'label', 'nine_partition_label', 'three_partition_label', 'qc', 'url', 'url_alphanum']]

def download_image(row, failed_images, progress):
    url = row['url']
    image_name = f"{row.md5hash}.jpg"  
    image_path = os.path.join(output_folder, image_name)

    try:
        subprocess.run(
            ["curl", "-L", "-o", image_path, url], 
            check=True, 
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        # print(f'Downloaded and saved as: {image_name}')
    except subprocess.CalledProcessError as e:
        failed_images.append(image_name)
        # print(f'Failed to download {url}: {e}')
    
    progress[0] += 1

    if progress[0] % 10 == 0:
        print(progress, ': ')
        print('succeeded: ', progress[0] - len(failed_images))
        print('failed: ', len(failed_images))

failed_images = []
progress = [0]
df['img_name'] = df.apply(lambda row: download_image(row, failed_images, progress), axis=1)

print('df length: ', len(df))
print('failed_images length: ', len(failed_images))

# saving truncated csv file
df.to_csv("/teamspace/studios/this_studio/fitzpatrick17k_trunc.csv", index=False)
     
# saving failed_images
dict = {'failed': failed_images}
df = pd.DataFrame(dict)
df.to_csv('failed_images.csv')