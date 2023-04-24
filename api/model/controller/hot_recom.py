import requests

from liontk.sql.pgsql import PGSQLMgr
from liontk.enum.pgsql import PGSQL


class HotRecom():
    @staticmethod
    def get_labels(tourid):
        try:
            pg_mgr = PGSQLMgr.get_mgr(PGSQL.DEV_DS)
            sql_str = f"SELECT labels FROM grp_tourid_labels WHERE tourid = '{tourid}';"
            result = list(pg_mgr.query(sql_str, to_dict=True))

        except Exception as e:
            print(e)

        finally:
            pg_mgr.close()

        if result:
            labels = [label.split('-')[-1]
                      for label, _ in sorted(result[0]['labels'].items(), key=lambda x: x[1], reverse=True)][:6]
            return labels
        else:
            return None

    def get_grp_info(self, grp_content):
        for grp in grp_content:
            grp_info = grp['plist'][0]
            labels = self.get_labels(grp_info['id'])

            if labels:
                yield {'tour_id': grp_info['id'],
                       'prod_name': grp_info['title'],
                       'prod_url': grp_info['url'],
                       'prod_price': grp_info['price'],
                       'img_url': grp_info['img'],
                       'labels': labels}
            else:
                continue

    def recommend(self):
        hot_api = 'http://10.35.2.45/product-tag/api/recommend/hot/?theme=%E6%B8%AC%E8%A9%A6&use_type=reco.system&sub_use_type=organic&recommend_product_type=grp&use_utm=0&rate=0'
        response = requests.get(hot_api).json()

        hot_grp = list()
        TOP = 5
        for grp_info in self.get_grp_info(response['Data']['results'][0]['json_content']):
            hot_grp.append(grp_info)
            if len(hot_grp) == TOP:
                break

        return hot_grp
