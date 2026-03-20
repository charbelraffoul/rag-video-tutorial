import weaviate
client = weaviate.connect_to_local(host="127.0.0.1", port=8080)
print([c.name for c  in client.collections.list_all()])  # should include "Scene"
client.close()
