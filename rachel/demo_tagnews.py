import tagnews

### Must run this from terminal to train tag model first
### Also must copy data from lib/data/ci-data into lib/data
### cd lib
### python -m tagnews.crimetype.models.binary_stemmed_logistic.save_model
crimetags = tagnews.CrimeTags()
article_text = ('The homicide occurred at the 1700 block of S. Halsted Ave.'
                ' It happened just after midnight. Another person was killed at the'
                ' intersection of 55th and Woodlawn, where a lone gunman')
print(crimetags.tagtext_proba(article_text))
# HOMI     0.739159
# VIOL     0.146943
# GUNV     0.134798
print(crimetags.tagtext(article_text, prob_thresh=0.5))
# ['HOMI']

### Must run this from terminal to train geo model first
### cd lib
### python -m tagnews.geoloc.models.lstm.save_model
geoextractor = tagnews.GeoCoder()
prob_out = geoextractor.extract_geostring_probs(article_text)
print(list(zip(*prob_out)))
# [..., ('at', 0.0044685714), ('the', 0.005466637), ('1700', 0.7173856),
#  ('block', 0.81395197), ('of', 0.82227415), ('S.', 0.7940061),
#  ('Halsted', 0.70529455), ('Ave.', 0.60538065), ...]
geostrings = geoextractor.extract_geostrings(article_text, prob_thresh=0.5)
print(geostrings)
# [['1700', 'block', 'of', 'S.', 'Halsted', 'Ave.'], ['55th', 'and', 'Woodlawn,']]
coords, scores = geoextractor.lat_longs_from_geostring_lists(geostrings)
print(coords)
#          lat       long
# 0  41.859021 -87.646934
# 1  41.794816 -87.597422
print(scores)  # confidence in the lat/longs as returned by pelias, higher is better
# array([0.878, 1.   ])
geoextractor.community_area_from_coords(coords)
# ['LOWER WEST SIDE', 'HYDE PARK']
