import io
import requests
import json
import torch
import sys
import numpy as np
import moviepy.editor as mpe

from model.pic_cls import zeroshot_cls
from fastapi.responses import StreamingResponse
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request, Response, Form
from typing import Any, Dict, Text, Tuple, List, Union
from io import BytesIO
from diffusers import StableDiffusionPipeline
from model.controller.img_recom import ImgRecom
from model.controller.hot_recom import HotRecom
from model.controller.popular_recom import PopularRecom
from model.controller.similar_recom import SimilarRecom
from moviepy.video import fx

sys.path.append(".")

torch_dtype = torch.float16
device_map = "auto"
torch.backends.cudnn.benchmark = True
pipe = StableDiffusionPipeline.from_pretrained(
    "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Anime-Chinese-v0.1", torch_dtype=torch.float16)
pipe.to('cuda')
pipe.safety_checker = lambda images, clip_input: (images, False)

ROOT_PATH = '/api/pic'
app = FastAPI(docs_url='{}/docs'.format(ROOT_PATH),
              openapi_url='{}/openapi.json'.format(ROOT_PATH))

search_grp_api = 'http://10.35.2.45/product-tag/api/es/search/group/?Countries=TW&WebCode=B2C&Keywords={}&SortType=3&TravelType=0&Page=1&PageSize=20&Language=zh_TW&IsAdvancedFilter=0&IsGetGroupList=0&IsPriceGt=1'


@app.get('{}/tourguide'.format(ROOT_PATH))
async def generate_tourguide_pic(prompt: Text):
    image = pipe(prompt, guidance_scale=7.5).images[0]
    img_bytes = BytesIO()
    image.save(img_bytes, format="png", quality=20)
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/png")


@app.post('{}/pic_cls'.format(ROOT_PATH))
async def create_upload_files(files: List[UploadFile] = File(...)):
    catlist = []
    imgs = []
    for file in files:
        img_bytes = await file.read()
        file.close()
        img = Image.open(io.BytesIO(img_bytes))
        imgs.append(img)
        result = zeroshot_cls(img)
        torch.cuda.empty_cache()
        catlist.append(result[0])

    img_recom = ImgRecom()
    recom = img_recom.recommend(search_grp_api, imgs, catlist)

    return {"catgorylist": catlist, "recommendation": recom}


@app.get('{}/hot_popular'.format(ROOT_PATH))
async def get_product(prompt: Text):
    if prompt == 'hot':
        hot_recom = HotRecom()
        return hot_recom.recommend()

    elif prompt == 'popular':
        pop_recom = PopularRecom()
        return pop_recom.recommend()


@app.get('{}/weather'.format(ROOT_PATH))
async def get_weather(lng: Text, lat: Text):
    weather_api = f'http://10.1.1.181/other/api/weather/latlon/?Lng={lng}&Lat={lat}&ElementName=MaxT%2CMinT%2CWx%2CPoP24h'
    res = requests.get(weather_api)
    result = json.loads(res.text)

    return result


@app.get('{}/product'.format(ROOT_PATH))
async def get_product(_id: Text):
    response = requests.get(search_grp_api.format(_id)).json()

    return response


@app.get('{}/similar'.format(ROOT_PATH))
async def get_product(_id: Text):
    response = requests.get(search_grp_api.format(_id)).json()
    similar_recom = SimilarRecom()

    return similar_recom.recommend(response['Data']['results'][0]['plist'][0]['TBM0_CODE'])


@app.get('{}/video'.format(ROOT_PATH))
async def video_endpoint(grp_id: Text):

    def iterfile(name):
        with open(f'src/anime_videos/{name}.mp4', mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile(grp_id), media_type="video/mp4")


@app.post('{}/files'.format(ROOT_PATH))
async def create_file(file: UploadFile = File(...), group_id: str = Form(...)):

    SIZE = (640, 480)
    img_bytes = await file.read()
    file.close()

    removebg_api_key = 'Tv59wQsigKYgiZr4v1GMoaPk'
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': img_bytes},
        data={'size': 'auto'},
        headers={'X-Api-Key': removebg_api_key},
    )
    if response.status_code == requests.codes.ok:
        img_bytes = response.content
    else:
        img_bytes = img_bytes

    img = Image.open(io.BytesIO(img_bytes))

    video = mpe.VideoFileClip(f"src/videos/{group_id}-result.mp4")
    img = mpe.ImageClip(np.array(img)).set_start(5).set_duration(int(video.duration) - 5).set_pos((0, int(SIZE[1]/1.3)))
    final = mpe.CompositeVideoClip([video, fx.all.resize(img, 0.2)])
    final.write_videofile(f"src/anime_videos/{str(group_id)}.mp4")

    if not file:
        return {"message": "No file sent"}
    else:
        return {"filename": file.filename}
