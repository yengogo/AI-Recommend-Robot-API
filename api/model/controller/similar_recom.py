import requests


class SimilarRecom():
    def recommend(self, norm_group_id):
        similar_api = f'http://10.35.2.45/product-tag/api/recommend/recently_viewed_featured_similar/?theme=%E6%B8%AC%E8%A9%A6&use_type=reco.system&sub_use_type=organic&channel=B2C&recommend_product_type=grp&search_count=10&use_utm=0&product_id={norm_group_id}&product_type=grp&countries=TW&is_price_gt=1'
        response = requests.get(similar_api).json()
        similar_grp = list()
        for grp in response['Data']['results'][0]['json_content']:
            grp_info = dict()
            grp_info['tour_id'] = grp['plist'][0]['id']
            grp_info['prod_name'] = grp['plist'][0]['title']
            grp_info['prod_url'] = grp['plist'][0]['url']
            grp_info['prod_price'] = grp['plist'][0]['price']
            grp_info['img_url'] = grp['plist'][0]['img']

            similar_grp.append(grp_info)

        return similar_grp
