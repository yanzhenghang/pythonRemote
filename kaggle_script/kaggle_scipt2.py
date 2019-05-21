import gzip
import base64
import os
from pathlib import Path
from typing import Dict




# this is base64 encoded source code
file_data: Dict = {'imet/transforms.py': 'H4sIADnwoVwC/61VyW7bMBC96yuI9EKlqiIbdpAa0aFNG9RAEBRpboFhsDYlEZVIl6LjJF/fGVIbbSftoQIEbm/eLOTMiGqjtCGaybWqAuFWFTNFEGRaVeT7/IY0u/OK5dztGqVXxaOohZKxAdk6U7qqWyANCHz36p7LWumI3MIhK8ULj8iVAkgNkzteuw0uDddXWm1gzxph55bBrb8pLV6UNKy8LsUmDIJgVbK6bk5/AMsaRWZW5OTkxO2TFewRU3CSi0cu0Y/YOgC2E9b4S9AGK6cyiwVVuZCstAcEMO5v4aze8BUGywh1VGwIiC0CiWaOrgHXEIuSa8LXudMtMAIbVaKQnJEvPGPb0sx6k+PP85v57ddPd62LgZ2seUaWSyGFWS5pzcssIi6mHmHqU7jIel8l5NIZnk7OphHc/lO7np5NXhPQnKVJPG7xuByFsw6M9sTW7bSPcnfgWQgIb+1De+sA1y/2QJ3JCOoWR5jAzpYHpkdYGkAzHUZ6BffWRVpU+cBbeP6EGcOrjQFf8LnknI6SAQK/hhtEbWQekgU57VejhQc2TOfctPa4BxhvpcBMo54/kW99CKSeb1a1jcfSPsy36Cws2g9qGHhkO3djVKutXFMsFnH9Wxs6NPnUUxmGocdQ/APD2QGDRyGy1gs30JBckiSezg7e6y6y+oqI7A4oduTSvxDM98LbHC0OKZ9GfRBxQF+SyCP6QHbhgdzzX+RGKFfseWpNrfLm5WBdo/RpFAFZhIa8RwefcSzCQ41QKTkEttVA0pRQDEh4RInmZqul1aJteaa0S+Ooz+gwOpLEA7p35Boy5Sdb/erTC3IHK4Gr+kNWv1Yd4205bEFPBw2jp+lBjQs2SFYpeo6+BtCm4IF3zQqZXC+iD4NmY3nHFxfh6x2IwtnCMvLa/GdC7JdvULYtlTZsXWOlFWcyfUjiyQUUZBim53ZIzhdwWWaNR+PxR9wbjydumC6c3j+kTaPZAggAAA==', 'imet/make_submission.py': 'H4sIADnwoVwC/22TTWvkMAyG7/kVvjnDTt3dPQ7kMLD3LmVupRhNLM+4JHawnNLur1/Z+YQ2hIRIj/K+tmTXDyEmAfE2QCSsKjcFBvAGSPA9mKqyMfRCjcl1JGagR/Da2DllIAFhWpJ/zpezfn56uszpHpxfclfnIbp/qIeIxrXJBV9VlUErMlUfTpXgq7iJolmdqXO8jT369Ldk6kPBOMvMBCswRsNMLdlabjIkj4K1b9TIH/KwEWFMw5j2kYeHdI9I99AZrkmfAza2C5COgn3C2KXmp/q98rRZKK9sgmaDBP3Qoabx2jsiNpFZoyKC0S291wXK17pl4lHIL1WKWXbivMEP3Yaukc7Mho3N+i+v5cOGKLYFM1/8qd0enFZFY3de7sbWG/W90lpkLBcVov7qtMT3BaRgGNCb2tjF8STcBt9C4jDt4vNc7eCX0yvHvxkbZtQ7dCMS92jpV1MWvH7u/sw0G+k+6xsm3XZAlAvhw1Hza8GUhx6ZlZBSdNcxoXaG5JJNoTStSExTcxR33j6MzSWOeJgHeSdQu4T9PNMR0xi9kEKqt8Cj3vJpyg3jN2845ZURj27uWq5S+cFzJJzdZbMGB7TOTrUWDZvVOh8dreWkM52j6j8RT2H62wMAAA==', 'imet/models.py': 'H4sIADnwoVwC/71Vy2rcMBTd+yu0iwyuYk8yIRmYRSCdLJqkpe2uFKHa1xmBR3YlOQkt/fde+SU7MykOtDUYLOmcc99yrssdyWuV2rIsDJG7qtSWVEJbKYog6Na21Ok2yB22+exxSo02mVL9fiMoSyUKIgzZTGQepMETtiszQIN4fBu0Iqy20rvw/o6/u7y+vnkbBEFaCGPI5cP9B3SSKsVuy6wuIFwFBJ8McpKX+lHojBoo8og8dSfu0WBrrciGiYd7XiF/kdEnhDCzFRV8Way+hmjBaaQahAWuwFJ8eVqYiFRI10IqyFbkG5I7YZl7/4hQ2Rg3WEYRsiadFA2H/SZyrsQO/DHjzQbnA+oR5P3WGl4Ju0VcfsTYsVRVbY9/eoFfkwWr7PZobJ4Vpci4sS6sTKaWtnVyu3SsH7bOYT3gJfd9gGv/2dK6DCNyKNVHMHcujQcqxblU0nLelUrVO95wwESD6eEZWd0I9C7q/VnfMg0GF8s4Ipkuq7K2LWRUelNXoGnIBpO+Bs44ayP8c9n3oh2zXU+5lkKVvjk9CHukc2w1CWwg56nLr2Kf4HsNys0b3csAHl+1IjSMDp3eoGdC05Eok4rnGFKNCZrk95mA93Ra94M+vsZM4KcSz7bYYVrsTMPdn8uxYgMEC9pNy/zR7iXoUzg04BUoA/+8BTNnBZfJIvlvbddfYwd7bshmE4/MJehJ/Q4X2YP/dlFHyq8tLrY8uu7bo/PKFXkK2eBNUNQUFxHBC7IQKaw/6xrCQ1J99hw8ZA8SHt0XM/IH0DiMyJskfMkDH0tDfh4z7mH3tZdSco7E7h9K27tw/+ZKzsMOfnI6A35y2sOX8Qz4Mu7hSZzM8SZOBsJyMYewXITuvzlMwIjTT99Ls+JpZxdzaWcXnraI51pbxBNrs508Q9pv8f1VdhsJAAA=', 'imet/__init__.py': 'H4sIADnwoVwC/8vMLcgvKlFILjPi4uICknrFqSV+pbkhGUWpiSnFGgaaCgrKCmmZFQoF+SWpeSWZiTkKBZUl+UXJGQrl+UXZqUUKmcXFpanFXABAVNvYSQAAAA==', 'imet/make_folds.py': 'H4sIADnwoVwC/31UTY/TMBC951f4FgdSoyK4VDJSxQqJC4vQ3qLK8sZOsTaxgz1B7L9n/JFuW7pEVRPPx5t5zx6baXYeiPTHWfqgq8G7ifRuHHUPxtlATA5QepDLCMr00JLPbrGgfVV8XlrlpmpdzriUgeBvVqsNfikMSOBMSZBBw4p8t3/Yix/39w9VVWEVMsknLQY3qkBtfu+IsdCQzScEZHeY/cXLSe8qgo8aCI9mr6USffhNT3DkHanBS2MZmusmRfdjEH1sPmBWYUHRSAbn0SlD0MjYImpXSwBvHhfQwqhQH1gAz8I8GqAZ6+aTcRJGgcvBkYe4qH4mKI30TmHR2W22B/KGjNpSNawuTwzoKUJHNVn8Qy8LcppHTQcve75ty2aIABI0//C+YZjkYcGQQJv29c4JOJAjLyWb3Sky0uFkMpbG8uxCllWQljzpZz7K6VHJmLA7U7rDz8OLZFmJVYWODu21OB2aUk7iPETCSOqo1+PQHE5g2FVOKh12eRHzBIKkBaaf1TzrJCvFgtYqU/tqlf5z2SnilrD+pzO9pt2Qu7qN/h9542OGksX5S+tX2oTupZcDlo+26sbxen0zdtXlibwSF9dZX/KWk20Zoq6O9notmIl4DYuPw3CaTJS4wKfbwmP4enOwvT8uk7bwPXnKkOQwJpUSsvhpvdnYTapStwSeZ83x/LfrQPCPORPD4wEpAOkVIUIBTnP/71XBYwhbD0qJZODS1ZA4hnQdtCghCsy/OasbpIc7I4TFW0WIuDm1EJGsEHVmm5lXfwGNKmW3LQUAAA==', 'imet/dataset.py': 'H4sIADnwoVwC/71VXWvbMBR996/Qy7BcXKXtYDBDB2nahrLQlDRvIRjFlj2BbBlJCetK//uubDn+2FfYQ02Ibenec889OpYyJQtUUfNN8B3iRSWVQU/w6mV2wrxUvMzb8RkVgu4EC9GCa+N5bjg5XLWPFS1TqhH8qrRBeHpYtOkPBc1ZG2mkStoi9pHsDReapNTQNv4WnjWDOnUUMYqWOpOq0G2AYaWWKj5OuMAaqY1ZPsZfp/P54s7zvMd4tpg+P989o2t0eXnx0budrqfxarlcw4BtGvuETHhZ7c2EF8ycX11cfj7P8kPyyUc867AQE5ohn0wsXT8A6ERQrdFaUV462tjdg8hDcKUsQ3HMS27iGGsmshApKU1U1w1RmkUgGbE594oWLKyTBhe38nXNRr3lSNlun0doJ6WATtZqz1xVe+l9xRQOyLF60E0BDxJbHpBmb6OZNIPxNBuNjohAyGhkjGLZWSB793pqCFY6MXp0FTN7VSKYwy2JoJ+UM8MNK44q8vR7hHhpehB2Huq16YQLmWwgbusNtIQQIWna8W46wwPpLVbYEyr8vQhuDa57HXcyG6qANZRrnP6DKanx0YxdHAChxFq3rMsSaoziu71hMU810ZXgBvfa7KA30D+GzGBrnT2WsqYautDOq+upc+g7GNQYOl6k9/EeFLa6G/pfvkNnHcw/PHiC/dCHIfqJdjzRgcEflr12Ek9h3S33vxu+KdVf9pM3nXsKW6ITYdBOvwmLHAxCRvi4fnchmSsxlIloemDYb74xUpW534S7rsdnQovYa3/MqOk1QOdfmiOK1P/DVuCMI7xQjKZYG4Vr205Q5r86ed8aJsEvScnBzKSQCrvlsGOz5WK5im/mq6vV/GbAvqlvjzGqFH0ZkgfX2Y0Ajzjbs3gDpLZRHwlEMCzFrxUp4Wt1m4cf+8HmYltvNJXdZiwUyYXcYf+s6eAt8H4CN4yOsxMIAAA=', 'imet/utils.py': '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', 'imet/main.py': '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', 'setup.py': 'H4sIADnwoVwC/0srys9VKE4tKS0oyc/PKVbIzC3ILyqBiHBxgSkNLgUgyEvMTbVVz8xNLVHXAQsUJCZnJ6anFttGQ0Rjdbg0uQA9nF1NTwAAAA=='}


# for path, encoded in file_data.items():
#     print(path)
#     path = Path(path)
#     path.parent.mkdir(exist_ok=True)
#     path.write_bytes(gzip.decompress(base64.b64decode(encoded)))

pythonVersion = 'python3'
def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:./kaggle/working && ' + command)


# run(pythonVersion+' setup.py develop --install-dir ./kaggle/working')
# run(pythonVersion+' -m imet.make_folds')
run(pythonVersion+' -m imet.main train model_2 --n-epochs 25 --model densenet169 --batch-size 32')
run(pythonVersion+' -m imet.main predict_test model_2 --model densenet169 --batch-size 32')
run(pythonVersion+' -m imet.make_submission model_2/test.h5 submission2.csv')









#-------------------------------------------------------------------------------------

import gzip
import base64
import os
from pathlib import Path
from typing import Dict




# this is base64 encoded source code
file_data: Dict = {'imet/transforms.py': 'H4sIADnwoVwC/61VyW7bMBC96yuI9EKlqiIbdpAa0aFNG9RAEBRpboFhsDYlEZVIl6LjJF/fGVIbbSftoQIEbm/eLOTMiGqjtCGaybWqAuFWFTNFEGRaVeT7/IY0u/OK5dztGqVXxaOohZKxAdk6U7qqWyANCHz36p7LWumI3MIhK8ULj8iVAkgNkzteuw0uDddXWm1gzxph55bBrb8pLV6UNKy8LsUmDIJgVbK6bk5/AMsaRWZW5OTkxO2TFewRU3CSi0cu0Y/YOgC2E9b4S9AGK6cyiwVVuZCstAcEMO5v4aze8BUGywh1VGwIiC0CiWaOrgHXEIuSa8LXudMtMAIbVaKQnJEvPGPb0sx6k+PP85v57ddPd62LgZ2seUaWSyGFWS5pzcssIi6mHmHqU7jIel8l5NIZnk7OphHc/lO7np5NXhPQnKVJPG7xuByFsw6M9sTW7bSPcnfgWQgIb+1De+sA1y/2QJ3JCOoWR5jAzpYHpkdYGkAzHUZ6BffWRVpU+cBbeP6EGcOrjQFf8LnknI6SAQK/hhtEbWQekgU57VejhQc2TOfctPa4BxhvpcBMo54/kW99CKSeb1a1jcfSPsy36Cws2g9qGHhkO3djVKutXFMsFnH9Wxs6NPnUUxmGocdQ/APD2QGDRyGy1gs30JBckiSezg7e6y6y+oqI7A4oduTSvxDM98LbHC0OKZ9GfRBxQF+SyCP6QHbhgdzzX+RGKFfseWpNrfLm5WBdo/RpFAFZhIa8RwefcSzCQ41QKTkEttVA0pRQDEh4RInmZqul1aJteaa0S+Ooz+gwOpLEA7p35Boy5Sdb/erTC3IHK4Gr+kNWv1Yd4205bEFPBw2jp+lBjQs2SFYpeo6+BtCm4IF3zQqZXC+iD4NmY3nHFxfh6x2IwtnCMvLa/GdC7JdvULYtlTZsXWOlFWcyfUjiyQUUZBim53ZIzhdwWWaNR+PxR9wbjydumC6c3j+kTaPZAggAAA==', 'imet/make_submission.py': 'H4sIADnwoVwC/22TTWvkMAyG7/kVvjnDTt3dPQ7kMLD3LmVupRhNLM+4JHawnNLur1/Z+YQ2hIRIj/K+tmTXDyEmAfE2QCSsKjcFBvAGSPA9mKqyMfRCjcl1JGagR/Da2DllIAFhWpJ/zpezfn56uszpHpxfclfnIbp/qIeIxrXJBV9VlUErMlUfTpXgq7iJolmdqXO8jT369Ldk6kPBOMvMBCswRsNMLdlabjIkj4K1b9TIH/KwEWFMw5j2kYeHdI9I99AZrkmfAza2C5COgn3C2KXmp/q98rRZKK9sgmaDBP3Qoabx2jsiNpFZoyKC0S291wXK17pl4lHIL1WKWXbivMEP3Yaukc7Mho3N+i+v5cOGKLYFM1/8qd0enFZFY3de7sbWG/W90lpkLBcVov7qtMT3BaRgGNCb2tjF8STcBt9C4jDt4vNc7eCX0yvHvxkbZtQ7dCMS92jpV1MWvH7u/sw0G+k+6xsm3XZAlAvhw1Hza8GUhx6ZlZBSdNcxoXaG5JJNoTStSExTcxR33j6MzSWOeJgHeSdQu4T9PNMR0xi9kEKqt8Cj3vJpyg3jN2845ZURj27uWq5S+cFzJJzdZbMGB7TOTrUWDZvVOh8dreWkM52j6j8RT2H62wMAAA==', 'imet/models.py': 'H4sIADnwoVwC/71Vy2rcMBTd+yu0iwyuYk8yIRmYRSCdLJqkpe2uFKHa1xmBR3YlOQkt/fde+SU7MykOtDUYLOmcc99yrssdyWuV2rIsDJG7qtSWVEJbKYog6Na21Ok2yB22+exxSo02mVL9fiMoSyUKIgzZTGQepMETtiszQIN4fBu0Iqy20rvw/o6/u7y+vnkbBEFaCGPI5cP9B3SSKsVuy6wuIFwFBJ8McpKX+lHojBoo8og8dSfu0WBrrciGiYd7XiF/kdEnhDCzFRV8Way+hmjBaaQahAWuwFJ8eVqYiFRI10IqyFbkG5I7YZl7/4hQ2Rg3WEYRsiadFA2H/SZyrsQO/DHjzQbnA+oR5P3WGl4Ju0VcfsTYsVRVbY9/eoFfkwWr7PZobJ4Vpci4sS6sTKaWtnVyu3SsH7bOYT3gJfd9gGv/2dK6DCNyKNVHMHcujQcqxblU0nLelUrVO95wwESD6eEZWd0I9C7q/VnfMg0GF8s4Ipkuq7K2LWRUelNXoGnIBpO+Bs44ayP8c9n3oh2zXU+5lkKVvjk9CHukc2w1CWwg56nLr2Kf4HsNys0b3csAHl+1IjSMDp3eoGdC05Eok4rnGFKNCZrk95mA93Ra94M+vsZM4KcSz7bYYVrsTMPdn8uxYgMEC9pNy/zR7iXoUzg04BUoA/+8BTNnBZfJIvlvbddfYwd7bshmE4/MJehJ/Q4X2YP/dlFHyq8tLrY8uu7bo/PKFXkK2eBNUNQUFxHBC7IQKaw/6xrCQ1J99hw8ZA8SHt0XM/IH0DiMyJskfMkDH0tDfh4z7mH3tZdSco7E7h9K27tw/+ZKzsMOfnI6A35y2sOX8Qz4Mu7hSZzM8SZOBsJyMYewXITuvzlMwIjTT99Ls+JpZxdzaWcXnraI51pbxBNrs508Q9pv8f1VdhsJAAA=', 'imet/__init__.py': 'H4sIADnwoVwC/8vMLcgvKlFILjPi4uICknrFqSV+pbkhGUWpiSnFGgaaCgrKCmmZFQoF+SWpeSWZiTkKBZUl+UXJGQrl+UXZqUUKmcXFpanFXABAVNvYSQAAAA==', 'imet/make_folds.py': 'H4sIADnwoVwC/31UTY/TMBC951f4FgdSoyK4VDJSxQqJC4vQ3qLK8sZOsTaxgz1B7L9n/JFuW7pEVRPPx5t5zx6baXYeiPTHWfqgq8G7ifRuHHUPxtlATA5QepDLCMr00JLPbrGgfVV8XlrlpmpdzriUgeBvVqsNfikMSOBMSZBBw4p8t3/Yix/39w9VVWEVMsknLQY3qkBtfu+IsdCQzScEZHeY/cXLSe8qgo8aCI9mr6USffhNT3DkHanBS2MZmusmRfdjEH1sPmBWYUHRSAbn0SlD0MjYImpXSwBvHhfQwqhQH1gAz8I8GqAZ6+aTcRJGgcvBkYe4qH4mKI30TmHR2W22B/KGjNpSNawuTwzoKUJHNVn8Qy8LcppHTQcve75ty2aIABI0//C+YZjkYcGQQJv29c4JOJAjLyWb3Sky0uFkMpbG8uxCllWQljzpZz7K6VHJmLA7U7rDz8OLZFmJVYWODu21OB2aUk7iPETCSOqo1+PQHE5g2FVOKh12eRHzBIKkBaaf1TzrJCvFgtYqU/tqlf5z2SnilrD+pzO9pt2Qu7qN/h9542OGksX5S+tX2oTupZcDlo+26sbxen0zdtXlibwSF9dZX/KWk20Zoq6O9notmIl4DYuPw3CaTJS4wKfbwmP4enOwvT8uk7bwPXnKkOQwJpUSsvhpvdnYTapStwSeZ83x/LfrQPCPORPD4wEpAOkVIUIBTnP/71XBYwhbD0qJZODS1ZA4hnQdtCghCsy/OasbpIc7I4TFW0WIuDm1EJGsEHVmm5lXfwGNKmW3LQUAAA==', 'imet/dataset.py': 'H4sIADnwoVwC/71VXWvbMBR996/Qy7BcXKXtYDBDB2nahrLQlDRvIRjFlj2BbBlJCetK//uubDn+2FfYQ02Ibenec889OpYyJQtUUfNN8B3iRSWVQU/w6mV2wrxUvMzb8RkVgu4EC9GCa+N5bjg5XLWPFS1TqhH8qrRBeHpYtOkPBc1ZG2mkStoi9pHsDReapNTQNv4WnjWDOnUUMYqWOpOq0G2AYaWWKj5OuMAaqY1ZPsZfp/P54s7zvMd4tpg+P989o2t0eXnx0budrqfxarlcw4BtGvuETHhZ7c2EF8ycX11cfj7P8kPyyUc867AQE5ohn0wsXT8A6ERQrdFaUV462tjdg8hDcKUsQ3HMS27iGGsmshApKU1U1w1RmkUgGbE594oWLKyTBhe38nXNRr3lSNlun0doJ6WATtZqz1xVe+l9xRQOyLF60E0BDxJbHpBmb6OZNIPxNBuNjohAyGhkjGLZWSB793pqCFY6MXp0FTN7VSKYwy2JoJ+UM8MNK44q8vR7hHhpehB2Huq16YQLmWwgbusNtIQQIWna8W46wwPpLVbYEyr8vQhuDa57HXcyG6qANZRrnP6DKanx0YxdHAChxFq3rMsSaoziu71hMU810ZXgBvfa7KA30D+GzGBrnT2WsqYautDOq+upc+g7GNQYOl6k9/EeFLa6G/pfvkNnHcw/PHiC/dCHIfqJdjzRgcEflr12Ek9h3S33vxu+KdVf9pM3nXsKW6ITYdBOvwmLHAxCRvi4fnchmSsxlIloemDYb74xUpW534S7rsdnQovYa3/MqOk1QOdfmiOK1P/DVuCMI7xQjKZYG4Vr205Q5r86ed8aJsEvScnBzKSQCrvlsGOz5WK5im/mq6vV/GbAvqlvjzGqFH0ZkgfX2Y0Ajzjbs3gDpLZRHwlEMCzFrxUp4Wt1m4cf+8HmYltvNJXdZiwUyYXcYf+s6eAt8H4CN4yOsxMIAAA=', 'imet/utils.py': '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', 'imet/main.py': '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', 'setup.py': 'H4sIADnwoVwC/0srys9VKE4tKS0oyc/PKVbIzC3ILyqBiHBxgSkNLgUgyEvMTbVVz8xNLVHXAQsUJCZnJ6anFttGQ0Rjdbg0uQA9nF1NTwAAAA=='}


# for path, encoded in file_data.items():
#     print(path)
#     path = Path(path)
#     path.parent.mkdir(exist_ok=True)
#     path.write_bytes(gzip.decompress(base64.b64decode(encoded)))

pythonVersion = 'python3'
def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:./kaggle/working && ' + command)


# run(pythonVersion+' setup.py develop --install-dir ./kaggle/working')
# run(pythonVersion+' -m imet.make_folds')
# run(pythonVersion+' -m imet.main train model_1 --n-epochs 25 --batch-size 16')
# run(pythonVersion+' -m imet.main predict_test model_1 --batch-size 16')
# run(pythonVersion+' -m imet.make_submission model_1/test.h5 submission.csv')

#
# run(pythonVersion+' -m imet.main train model_1 --n-epochs 25 --model resnet101 --batch-size 32 --fold 3')
# run(pythonVersion+' -m imet.main predict_test model_1 --model resnet101 --batch-size 32 --fold 3')
# run(pythonVersion+' -m imet.make_submission model_1/test.h5 submission.csv')

# run(pythonVersion+' setup.py develop --install-dir ./kaggle/working')
# run(pythonVersion+' -m imet.make_folds --n-folds 40')
run(pythonVersion+' -m imet.main train model_1 --n-epochs 17 --model resnet101 --batch-size 32 --fold 10 --patience 2 --lr 0.0001')
run(pythonVersion+' -m imet.main predict_test model_1 --model resnet101 --batch-size 32 --fold 10 --patience 2 --lr 0.0001')
run(pythonVersion+' -m imet.make_submission model_1/test.h5 submission.csv')





