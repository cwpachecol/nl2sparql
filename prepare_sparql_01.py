from parsers.lc_quad_linked import LC_Qaud_Linked
from parsers.lc_quad_linked import LC_Qaud_Linked

ds = LC_Qaud_Linked(path="./data/LC-QuAD10/linked_answer.json")
ds.load()
ds.parse()
for indx, qapair in enumerate(ds.qapairs[4900:]):
    print(f"No: {indx}\n Uri's: {qapair.sparql.query}")
    # print(f"{qapair.sparql.where_clause_template}..")
    # print(f"{qapair.sparql.raw_query}..")
    # print(f"{qapair.sparql.query_features()}..")

    new_sparql = str(qapair.sparql.query)
    print("--"*50)
    for uri in qapair.sparql.uris:
        print(f"{uri.raw_uri} -- {uri.uri_type}")
        if uri.uri_type == "?s":
            new_sparql = new_sparql.replace(str(uri.raw_uri), str(uri.uri_type))

    print(new_sparql)
    print("+++++"*30)