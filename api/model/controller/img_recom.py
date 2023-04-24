import numpy as np
import requests
import torch
import pandas as pd

from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from scipy import stats
from liontk.sql.pgsql import PGSQLMgr
from liontk.enum.pgsql import PGSQL

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ImgRecom():
    def __init__(self) -> None:
        self.model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").to(device)
        self.processor = r = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

    @staticmethod
    def search_db(img_labels):
        try:
            pg_mgr = PGSQLMgr.get_mgr(PGSQL.DEV_DS)
            syntax1 = [f"COALESCE((labels->>'{label}')::int, 0)" for label in img_labels]
            syntax2 = [f"labels ? '{label}'" for label in img_labels]
            sql_str = f"""
            SELECT *,{' + '.join(syntax1)} AS total
                            FROM grp_tourid_labels
                            WHERE {' OR '.join(syntax2)}
                            ORDER BY total DESC;"""
            result = list(pg_mgr.query(sql_str, to_dict=True))

        except Exception as e:
            print(e)

        finally:
            pg_mgr.close()

        df = pd.DataFrame(result)

        return df

    def images_to_tensors(self, imgs):
        img_tensors = list()
        for img in imgs:
            inputs = self.processor(images=img, return_tensors="pt").to(device)

            with torch.no_grad():
                img_tensors.append(self.model.get_image_features(**inputs))

        return img_tensors

    @staticmethod
    def get_info_by_api(search_grp_api, df, img_labels, img_tensors):
        count = list()
        cos_sim_mean = list()
        click_count_mean = list()
        selling_ratio_mean = list()
        earliest_departure_group_of_tourid = list()
        for tourid, labels_with_pic_no in zip(df['tourid'].values, df['labels_with_pic_no'].values):
            response = requests.get(search_grp_api.format(tourid)).json()
            if response.get('Data'):
                labels = [label.split('-')[-1]
                          for label, _ in sorted(labels_with_pic_no.items(), key=lambda x: len(x[1]), reverse=True)][:6]

                cos_sim_values = list()
                for img_label, img_tensor in zip(img_labels, img_tensors):

                    if labels_with_pic_no.get(img_label):
                        if img_label.split('-')[-1] in labels:
                            labels.pop(labels.index(img_label.split('-')[-1]))
                            labels.insert(0, img_label.split('-')[-1])
                        else:
                            labels.pop()
                            labels.insert(0, img_label.split('-')[-1])

                        for pic_no in labels_with_pic_no[img_label]:

                            if '-' in pic_no:
                                folder = f"src/image_tensors/cms_imgs/{pic_no.split('-')[1][:3]}"
                                tensor = torch.load(f"{folder}/{pic_no}.pt")
                                cos_sim = torch.nn.functional.cosine_similarity(img_tensor, tensor).item()
                                cos_sim_values.append(cos_sim)

                            elif '_' in pic_no:
                                folder = f"src/image_tensors/non_cms_imgs/{pic_no.split('_')[1][:3]}"
                                tensor = torch.load(f"{folder}/{pic_no}.pt")
                                cos_sim = torch.nn.functional.cosine_similarity(img_tensor, tensor).item()
                                cos_sim_values.append(cos_sim)

                response_info = response['Data']['results'][0]['plist'][0]
                norm_group_id = response_info['TBM0_CODE']
                try:
                    img_url = response_info['IMAGEURL']
                except Exception:
                    img_url = 'https://uwww.liontravel.com//cto/view/default16-9.jpg'

                group_info = response_info['GROUP_INFO']

                click_count_values = list()
                selling_ratio_values = list()
                group_id = ''
                name = ''
                price = ''
                for info in group_info:
                    click_count_values.append(info['CLICK_COUNT'])
                    selling_ratio_values.append(1-info['QUOTA_SEATS']/info['TOTAL_SEATS'])

                    if not group_id:
                        group_id = info['PROD_NO']
                        name = response_info['TOUR_INSIDE_NAME']
                        price = info['B2C_LOW_PRICE']

                count.append(len(cos_sim_values))
                cos_sim_mean.append(np.mean(cos_sim_values))
                click_count_mean.append(np.mean(click_count_values))
                selling_ratio_mean.append(np.mean(selling_ratio_values))
                earliest_departure_group_of_tourid.append(
                    [group_id, norm_group_id, name, price, img_url, labels])

        return count, cos_sim_mean, click_count_mean, selling_ratio_mean, earliest_departure_group_of_tourid

    @ staticmethod
    def rank_info(api_info, TOP):
        count = api_info[0]
        cos_sim_mean = api_info[1]
        click_count_mean = api_info[2]
        selling_ratio_mean = api_info[3]
        earliest_departure_group_of_tourid = api_info[4]

        scores_of_alternatives = list()
        for c, c_s, c_c, s_r in zip(stats.zscore(count),
                                    stats.zscore(cos_sim_mean),
                                    stats.zscore(click_count_mean),
                                    stats.zscore(selling_ratio_mean)):
            scores_of_alternatives.append(0.4*c+0.3*c_s+0.15*c_c+0.15*s_r)

        sorted_lst = sorted(scores_of_alternatives, reverse=True)
        ranks = [sorted_lst.index(x) for x in scores_of_alternatives]

        recommendation = list()
        norm_group_id = list()
        for i in range(len(ranks)):

            if len(recommendation) == TOP:
                break

            group_info = earliest_departure_group_of_tourid[ranks.index(i)]

            if group_info[1] not in norm_group_id:
                norm_group_id.append(group_info[1])
                recom = dict()
                keys = ['group_id', 'prod_name', 'prod_price', 'img_url', 'labels']
                values = [content for idx, content in enumerate(group_info) if idx != 1]

                for key, value in zip(keys, values):
                    recom.setdefault(key, value)

                recommendation.append(recom)

        return recommendation

    def recommend(self, search_grp_api, imgs, img_labels):
        df = self.search_db(img_labels)

        with open('src/tourid_in_api.txt', 'r') as reader:
            tourid_in_api = reader.read().splitlines()

        df = df[df['tourid'].isin(tourid_in_api)]

        img_tensors = self.images_to_tensors(imgs)

        api_info = self.get_info_by_api(search_grp_api, df, img_labels, img_tensors)

        TOP = 5
        recommendation = self.rank_info(api_info, TOP)

        return recommendation
