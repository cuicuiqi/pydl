import pymysql

def getAllFAQ():
    FAQs = {}

    db = pymysql.connect(host='r.nbopenplatform.p.mysql.elong.com', port=6240, user='nbopenplatform_r',
                         password='', database='nb_open_platform')
    cursor = db.cursor()
    sql = "SELECT id,doc_name FROM nb_open_doc WHERE doc_type=3 AND plat_type=2 AND is_deleted=0"
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            id = row[0]
            doc_name = row[1]
            FAQs.update({id : doc_name})
    except Exception as e:
        print(e)

    db.close()
    return FAQs


def getAllFAQTest():
    FAQs = {114:'接口的默认访问频率'}
    return FAQs

