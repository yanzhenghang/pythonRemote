import gzip
import base64
import os
from pathlib import Path
from typing import Dict




# this is base64 encoded source code
file_data = ''
file_data: Dict = {'/home/yanzhenghang/pythonRemote/kaggle_script/imet/make_submission.py': 'H4sIAHFi2lwC/4VUTW/bMAy9+1cI2EHy1nrtsMMawIcAO/TWoshlCwJBsehEhW0Zolw0+/Wj5M+k7RYYMcz3SD6RFE3dWueZcodWOYQkMb2hVY1WyOhp9Wh7RtskpbM1of5YmT0bgEf6THok67ypcARqUI3U5QBp5RWCH8Gf681aPj08bAa4VqYZsb1plDN/QLYOtCm8ocxJoqFkgSU+1ydSnK4SRr8o3LF8OkS2doeuhsY/RkSkkUYocXpyprSWamBF1HWNdNZ6iT6E4jxaTcmkbFQNUrKcrFKG9FLyPvMQVvBZJfIrRtIPmPMvPD1n2c63nT+34qwpvoIqFDPlQldAs0W27c0uUqFCuNB0ff2BqitGdVRd5fNYRQqRXnoOSi+Zt7sPhY2hgqAkkj5RjUGz/Yn9+n3Pbu9uvn+7Oys1eYXBEctI25vV9Y8hS32i/ilihbnLKqs0ionMvjJOBVM1ZgHmaeZAaenh1Ys0TZLFYfzRAR5tpek8/tRCXlIsvzxbyLPle0AvN/dPfDcNzL+6g6puK5DY7WuDSEUOXN3LKPBFTJWa5jxofuOVEZeEmUbDqyxslXOjhxHRZci/7TtcWsfmhhL/zSzM/dflQstRl2JmvZ9pctIlOUWGeKs02pcOmKm2hUYLXY6K+8SFbQrlyYwL+7AMFuTtakf2d+46cbIXVXWA1LKxfXk88PS5iExsElKdxAG8LCqFGBzVq8H8dqRl4RqHm628d2bfeZBGIx9Rb2PTYop++q/YkcoHLt+4DtJh+ywSCOOhHlaQA9+5hnHGs2dL+6mgFRgaRm8qOIaTIW2a0LXglYU/mqOwX2Y05PjPwonLL03+AgCWDui1BQAA', '/home/yanzhenghang/pythonRemote/kaggle_script/imet/main.py': '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', '/home/yanzhenghang/pythonRemote/kaggle_script/imet/__init__.py': 'H4sIAHFi2lwC/8vMLcgvKlFILjPi4uICknrFqSV+pbkhGUWpiSnFGgaaCgrKCmmZFQoF+SWpeSWZiTkKBZUl+UXJGQrl+UXZqUUKmcXFpanFXABAVNvYSQAAAA==', '/home/yanzhenghang/pythonRemote/kaggle_script/imet/dataset.py': 'H4sIAHFi2lwC/71VbWvbMBD+7l8hCiNym8pJuw0a6CB9WSgLTUnzLRSjxLIrKktGUsLS0v++k1/il61b2YeaENnS3XPPPXeSYq1SlFH7KPgK8TRT2qI7+PRit2B3GZdJNX9JhaArwfpoyo31vHJ6vT2pXjMqI2oQ/LKoQLi7mVbuNylNWGVplV5XQdwr2VguDImopZX9FbwbBnFyK2I1lSZWOjWVgWXSKB3uF0rDHKmymd2GP8aTyfTa87zb8HI6vr+/vkfnaDgcnHrp7opr+DgIHlXKgh2Vz49MJo9UJsETTRLBgiemJRPBkAzPBp+HpyjTDOJxySI0Z0Yy+2WAjlCs1lQgoYwJDryr8WIczmezBSA7KXGPkIDLbGMDnjJ7fDIYnh3HyXb9tYd4XDNETBiGck5Hb7v4kMhaUGPQwvEoRcLl6I88BE/EYhSGXHIbhtgwEfeRVsqOcj59FMUjKBBxPt81TVk/d2o93BWrlnbUKH7EVptkhFZKCchwoTesjOoes8mYxj7ZR/frJeBBQscD3NzQWYlimI/izmyHCJh0Zroojp0DcqPXUEMwWYrRoAvV3GiJYA1XJPymU8Istyzdq8ijnyPEpW1AuHWIV7kTLtR6CXYPXktLMBGKRjXvIjPckt5h9RtC9f8sQlmD80bGtcyWamAN4Yp99cy0Mnjf+rUdAKG12ygyD0uotZqvNpaFPDLEZIJb3Eizhl5C/hg8/Qe3j7pS5lT7pWndq4tx2aEf0KDW0m6RPqb3ILDT3dL/6jt0WMP8owff0X7oUxv9ne34zg703yh73kk8gro77n9v+CJUs+zvPnS+UzgqSxFa6TSTcMh+y6SDj/Pv0iQuQ7RlIoZuGe4Ve4xkMukV5mXW3RuoQmyk32VU5Oqj42/FhUjy/3YqcKMSnmpGI2ysxnnbBijuvZTyvhZM/N+c1lt7qYTSuCyHm7ucTWfz8GIyP5lPLlrsi/ju0qRa012bPHSdOwhwh7O7+ZdA6mHURAIRLIvwS0Yk7Nby8OiFPX85eMgPmswdMw6KJEKtcO+wyODV934BRDIIcIEIAAA=', '/home/yanzhenghang/pythonRemote/kaggle_script/imet/make_folds.py': 'H4sIAHFi2lwC/61U34vUMBB+718RuIe22o23ogcuVDg8RF88kXvRsoRsM70L16Y1mRXvv3fSpN2fJwiWZdtMZr755stMdDf0Fpm094O0DpLG9h2r+7aFGnVvHNPBQUEjty0qXWPBPvRbg2CTuGelUX2XTMuBltIx+g1qsuFPRQ4jOFcSpQOckG+u767Ft9vbuyRJLphUChTbPLHvPz6x5bvLt5dXCeVmPkiAsbp+6MBgppoVwfMbMn+0soN8lTB6Bunc+HHB6taJ2hN1rJwYZ2RkTW9pkxyBqjNMNVUqEa3ebBGEVi5dc4eWu6HVmOUR7ewTkEaUCBjcLeDWemQqyZPv5COIpm+Vy0x4rygGc7Z4f1BEqEE1RJjMFqQStfuVzQqxVyxFK7XhZE4nauckm3FOdQth/1Odf9LGly8Osu+1VuZVmd38ZrVYrtkL1oKZqXtkjdB5aN9X3P/RLneyG1rIGivrclnEthQOJUL55nXOKcjillxclhfPM2fYo2zLmDI2VtSMGHXaZD49P5BlEqRgj/BUtrLbKOkDVntKV/S53kkWlJhUqLKmOBanItMYM9bc+IKpqHuYuihfz2DEKgRFhlVY+DhBIOOCwvdy7jEJSnEHoEJpn42C34dMCTe61Q+9riGrmsDqPPpf5PWPbmJUWe6oH2njqh2XNaX3tuRMez1/GKvksCOPxKV10Je9LNkyzkyVens6JXTPDDRJPF85dG9acp/uUH5t77d+2L6OO3FIghunYRUy7mfpYmEWY5a0YPg0QEn9X0wDUV6FSHL3DRIBxpeHcBF4HPPTG6b0LnxqlOjJsT+9UUaP8UYpSE4Su/zSG8ipVDolIQxdTEL4g0qF8IULkYbKgwrJH1RfB1xDBgAA', '/home/yanzhenghang/pythonRemote/kaggle_script/imet/utils.py': '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', '/home/yanzhenghang/pythonRemote/kaggle_script/imet/transforms.py': 'H4sIAHFi2lwC/81W247bNhB911cQ6xcqZbW2awcbI3pItw26QRAU26AvC8NgZMoiKokqSce7+frOkLrR9rrbPFWAIF7OnDkzw4tk1Shtieb1VlWR9L2K2yKKcq0q8vvdR9KO3lV8J/yoVTorvkojVZ1YsDW50pXpgDQi8HxWn0VtlGbkE0zyUn4TjNwqgBho3AvjB0Rthb7VqoExJ8K1HYPv/6a0/KZqy8v3pWziqBP5jAZuiNV5BE9WcmNakj/A2RaZV4756urKj5MMxogtBNnJr6LGcBMXJ9AT3qaFoFRnp3KHBUU7WfPSTRDA+LeDc9OIDHNqpTprNgYkDoFEK0/Xgg2krBSaiO3O+5aYqEaVaFSvyC8i5/vSrgbJyc93H+8+/fruvgsxco2tyMlmI2tpNxtqRJkz4lMfEKYhhS9A8FSy3njh6eJ6yWCRPHb95fXiOQMteDpN5h0eu7N41YNRT+LCTocs9xOBQkAE/RA6qAPc0DkC9ZIR1HfOMIHOjgeaZ1haQNscZzqDuvWZltVuFC2sUMKtFVVjIRZcLjtBZ9MRAp+WG0xdZh6ma/Jq6M3WAdhyvRO20+MXYLKvJW4GGsTDQvUxkAaxOdcuHxu3MC/RORg7TmocBWQHXzGq1b7eUjxTEvO3tnQs+VXgMo7jgKF4AcP1CUNAIfMuCv+hMXlLpslydbJeD8z5Kxg5nFAcyNuwILjfi2Bwtj6lfJwNScQPxjJlAdGP5BCf2D39i90M7YqjSJ3UateuHDzXKH2cMSBjKOQHDPAJv0V86hFOSgGJ7TyQNCUUExKfcaKF3evaedHuFKe038Zs2NExO7OJR3QT8h52yhee/TVsL9g7eBL4y2HMGp5V53g7Dnegp6N7ZaAZQG0ILknOKUaOsUYTuD04LPH+RkEuf2nRB5h1e07nib8+3jVN+UQfzl1UtIFzbxGz1ubkGUj+hMzLbGQ1f4nVvbIucrpYXobfqlLpDxKOHU2/wA1U2FoYA25gXWQgV3Njfa/YC2zEa0ZQxpue9+gOpZjMdH5zw8Zn/I2Dr2OXwk7lbaFkJroMOWOwi1lbYWyvwe5SzgcBzvyn6TR+/veAwtwa64gqhLEX6zj5rkpOvrOWk/9azcn/pJ4X8nhUG8fz0gJFQIz/hxeou19I2hL2P5K0ErxOH6AsN/BnAZ/la/eZvoZYjd3i1Hz+Bsfm84X/LNfe9T+S6gRs8goAAA==', '/home/yanzhenghang/pythonRemote/kaggle_script/imet/models.py': 'H4sIAHFi2lwC/71VXWvbMBR9z6/QW2XwVDttQhvIQ6FLH9Z2Y9vbGEKzrxuBI3uSnJaN/fdd+Ut2kw4XthkMlnTOud9yposdySqV2KLIDZG7stCWlEJbKfLZrF3bQifbWeaw9WeHU2qwyZTq9mtBWSiRE2HIZiSzlwZP2K5IAQ3i8d1s1qiwykrvw/t7/u7q5ub2LR4nuTCGXO0fPqCXVCl2V6RVDsFqRvBJISNZoR+FTqmBPAvJU3viHg220opsmNg/8BL585Q+IYSZrSjhy3z1NUALTiPRICxwBZbiy5PchKREuhZSQboi35DcCsvM+0eESoe43jKKkDVppWjQ79ehcyV24I8Zrzc471GPIB+21vBS2C3ishPGTqUqK3v60wv8Gi1YabcnQ/MsL0TKjXVhpTKxtCmU26VD/aBxDgsCL7nvA1z7z4bWZhiRfak+grl3aTxSKc6lkpbztlSq2vGaAybsTffPwOpGoHdh58/6jmkwuFhEIUl1URaVbSCD0puqBE0D1pv0NXDGWRPhn8t+EO2Q7XrKtRSqdM3pQdgjrWOrUWA9OUtcfhX7BN8rUG7g6EEG8Pi6EaFBeOz0Fj0Tmg5EmVQ8w5AqTNAov88EvKfjuh/18TVmZn4q8WyLHabFztTcw7kcKtZAsKDdtEwf7U6CPgV9A16DMvDPWzB1VnAZz+P/1nbdNXa05/ps1vHITIIe1e94kT34bxd1oPza4mLLo+u+PVqvXJHHkA3eBHlFcRESvCBzkcD6s64gOCbVZc/BA7aX8Oi+mJE/gEZBSN7EwUse+Fhq8vOYcQ+7r7mU4gsktj9R2tyFhzdXfBG08LPzCfCz8w6+iCbAF1EHj6N4ijdR3BMW8ymExTxw/81+AgacbvpemhVPW15OpS0vPW0eTbU2j0bWJju5RNpvZzM8bRwJAAA=', 'setup.py': 'H4sIAHFi2lwC/0srys9VKE4tKS0oyc/PKVbIzC3ILyqBiHBxgSkNLgUgyEvMTbVVz8xNLVHXAQsUJCZnJ6anFttGQ0Rjdbg0uQA9nF1NTwAAAA=='}
# 'imet/transforms.py': 'H4sIADnwoVwC/61VyW7bMBC96yuI9EKlqiIbdpAa0aFNG9RAEBRpboFhsDYlEZVIl6LjJF/fGVIbbSftoQIEbm/eLOTMiGqjtCGaybWqAuFWFTNFEGRaVeT7/IY0u/OK5dztGqVXxaOohZKxAdk6U7qqWyANCHz36p7LWumI3MIhK8ULj8iVAkgNkzteuw0uDddXWm1gzxph55bBrb8pLV6UNKy8LsUmDIJgVbK6bk5/AMsaRWZW5OTkxO2TFewRU3CSi0cu0Y/YOgC2E9b4S9AGK6cyiwVVuZCstAcEMO5v4aze8BUGywh1VGwIiC0CiWaOrgHXEIuSa8LXudMtMAIbVaKQnJEvPGPb0sx6k+PP85v57ddPd62LgZ2seUaWSyGFWS5pzcssIi6mHmHqU7jIel8l5NIZnk7OphHc/lO7np5NXhPQnKVJPG7xuByFsw6M9sTW7bSPcnfgWQgIb+1De+sA1y/2QJ3JCOoWR5jAzpYHpkdYGkAzHUZ6BffWRVpU+cBbeP6EGcOrjQFf8LnknI6SAQK/hhtEbWQekgU57VejhQc2TOfctPa4BxhvpcBMo54/kW99CKSeb1a1jcfSPsy36Cws2g9qGHhkO3djVKutXFMsFnH9Wxs6NPnUUxmGocdQ/APD2QGDRyGy1gs30JBckiSezg7e6y6y+oqI7A4oduTSvxDM98LbHC0OKZ9GfRBxQF+SyCP6QHbhgdzzX+RGKFfseWpNrfLm5WBdo/RpFAFZhIa8RwefcSzCQ41QKTkEttVA0pRQDEh4RInmZqul1aJteaa0S+Ooz+gwOpLEA7p35Boy5Sdb/erTC3IHK4Gr+kNWv1Yd4205bEFPBw2jp+lBjQs2SFYpeo6+BtCm4IF3zQqZXC+iD4NmY3nHFxfh6x2IwtnCMvLa/GdC7JdvULYtlTZsXWOlFWcyfUjiyQUUZBim53ZIzhdwWWaNR+PxR9wbjydumC6c3j+kTaPZAggAAA==', 'imet/make_submission.py': 'H4sIADnwoVwC/22TTWvkMAyG7/kVvjnDTt3dPQ7kMLD3LmVupRhNLM+4JHawnNLur1/Z+YQ2hIRIj/K+tmTXDyEmAfE2QCSsKjcFBvAGSPA9mKqyMfRCjcl1JGagR/Da2DllIAFhWpJ/zpezfn56uszpHpxfclfnIbp/qIeIxrXJBV9VlUErMlUfTpXgq7iJolmdqXO8jT369Ldk6kPBOMvMBCswRsNMLdlabjIkj4K1b9TIH/KwEWFMw5j2kYeHdI9I99AZrkmfAza2C5COgn3C2KXmp/q98rRZKK9sgmaDBP3Qoabx2jsiNpFZoyKC0S291wXK17pl4lHIL1WKWXbivMEP3Yaukc7Mho3N+i+v5cOGKLYFM1/8qd0enFZFY3de7sbWG/W90lpkLBcVov7qtMT3BaRgGNCb2tjF8STcBt9C4jDt4vNc7eCX0yvHvxkbZtQ7dCMS92jpV1MWvH7u/sw0G+k+6xsm3XZAlAvhw1Hza8GUhx6ZlZBSdNcxoXaG5JJNoTStSExTcxR33j6MzSWOeJgHeSdQu4T9PNMR0xi9kEKqt8Cj3vJpyg3jN2845ZURj27uWq5S+cFzJJzdZbMGB7TOTrUWDZvVOh8dreWkM52j6j8RT2H62wMAAA==', 'imet/models.py': 'H4sIADnwoVwC/71Vy2rcMBTd+yu0iwyuYk8yIRmYRSCdLJqkpe2uFKHa1xmBR3YlOQkt/fde+SU7MykOtDUYLOmcc99yrssdyWuV2rIsDJG7qtSWVEJbKYog6Na21Ok2yB22+exxSo02mVL9fiMoSyUKIgzZTGQepMETtiszQIN4fBu0Iqy20rvw/o6/u7y+vnkbBEFaCGPI5cP9B3SSKsVuy6wuIFwFBJ8McpKX+lHojBoo8og8dSfu0WBrrciGiYd7XiF/kdEnhDCzFRV8Way+hmjBaaQahAWuwFJ8eVqYiFRI10IqyFbkG5I7YZl7/4hQ2Rg3WEYRsiadFA2H/SZyrsQO/DHjzQbnA+oR5P3WGl4Ju0VcfsTYsVRVbY9/eoFfkwWr7PZobJ4Vpci4sS6sTKaWtnVyu3SsH7bOYT3gJfd9gGv/2dK6DCNyKNVHMHcujQcqxblU0nLelUrVO95wwESD6eEZWd0I9C7q/VnfMg0GF8s4Ipkuq7K2LWRUelNXoGnIBpO+Bs44ayP8c9n3oh2zXU+5lkKVvjk9CHukc2w1CWwg56nLr2Kf4HsNys0b3csAHl+1IjSMDp3eoGdC05Eok4rnGFKNCZrk95mA93Ra94M+vsZM4KcSz7bYYVrsTMPdn8uxYgMEC9pNy/zR7iXoUzg04BUoA/+8BTNnBZfJIvlvbddfYwd7bshmE4/MJehJ/Q4X2YP/dlFHyq8tLrY8uu7bo/PKFXkK2eBNUNQUFxHBC7IQKaw/6xrCQ1J99hw8ZA8SHt0XM/IH0DiMyJskfMkDH0tDfh4z7mH3tZdSco7E7h9K27tw/+ZKzsMOfnI6A35y2sOX8Qz4Mu7hSZzM8SZOBsJyMYewXITuvzlMwIjTT99Ls+JpZxdzaWcXnraI51pbxBNrs508Q9pv8f1VdhsJAAA=', 'imet/__init__.py': 'H4sIADnwoVwC/8vMLcgvKlFILjPi4uICknrFqSV+pbkhGUWpiSnFGgaaCgrKCmmZFQoF+SWpeSWZiTkKBZUl+UXJGQrl+UXZqUUKmcXFpanFXABAVNvYSQAAAA==', 'imet/make_folds.py': 'H4sIADnwoVwC/31UTY/TMBC951f4FgdSoyK4VDJSxQqJC4vQ3qLK8sZOsTaxgz1B7L9n/JFuW7pEVRPPx5t5zx6baXYeiPTHWfqgq8G7ifRuHHUPxtlATA5QepDLCMr00JLPbrGgfVV8XlrlpmpdzriUgeBvVqsNfikMSOBMSZBBw4p8t3/Yix/39w9VVWEVMsknLQY3qkBtfu+IsdCQzScEZHeY/cXLSe8qgo8aCI9mr6USffhNT3DkHanBS2MZmusmRfdjEH1sPmBWYUHRSAbn0SlD0MjYImpXSwBvHhfQwqhQH1gAz8I8GqAZ6+aTcRJGgcvBkYe4qH4mKI30TmHR2W22B/KGjNpSNawuTwzoKUJHNVn8Qy8LcppHTQcve75ty2aIABI0//C+YZjkYcGQQJv29c4JOJAjLyWb3Sky0uFkMpbG8uxCllWQljzpZz7K6VHJmLA7U7rDz8OLZFmJVYWODu21OB2aUk7iPETCSOqo1+PQHE5g2FVOKh12eRHzBIKkBaaf1TzrJCvFgtYqU/tqlf5z2SnilrD+pzO9pt2Qu7qN/h9542OGksX5S+tX2oTupZcDlo+26sbxen0zdtXlibwSF9dZX/KWk20Zoq6O9notmIl4DYuPw3CaTJS4wKfbwmP4enOwvT8uk7bwPXnKkOQwJpUSsvhpvdnYTapStwSeZ83x/LfrQPCPORPD4wEpAOkVIUIBTnP/71XBYwhbD0qJZODS1ZA4hnQdtCghCsy/OasbpIc7I4TFW0WIuDm1EJGsEHVmm5lXfwGNKmW3LQUAAA==', 'imet/dataset.py': 'H4sIADnwoVwC/71VXWvbMBR996/Qy7BcXKXtYDBDB2nahrLQlDRvIRjFlj2BbBlJCetK//uubDn+2FfYQ02Ibenec889OpYyJQtUUfNN8B3iRSWVQU/w6mV2wrxUvMzb8RkVgu4EC9GCa+N5bjg5XLWPFS1TqhH8qrRBeHpYtOkPBc1ZG2mkStoi9pHsDReapNTQNv4WnjWDOnUUMYqWOpOq0G2AYaWWKj5OuMAaqY1ZPsZfp/P54s7zvMd4tpg+P989o2t0eXnx0budrqfxarlcw4BtGvuETHhZ7c2EF8ycX11cfj7P8kPyyUc867AQE5ohn0wsXT8A6ERQrdFaUV462tjdg8hDcKUsQ3HMS27iGGsmshApKU1U1w1RmkUgGbE594oWLKyTBhe38nXNRr3lSNlun0doJ6WATtZqz1xVe+l9xRQOyLF60E0BDxJbHpBmb6OZNIPxNBuNjohAyGhkjGLZWSB793pqCFY6MXp0FTN7VSKYwy2JoJ+UM8MNK44q8vR7hHhpehB2Huq16YQLmWwgbusNtIQQIWna8W46wwPpLVbYEyr8vQhuDa57HXcyG6qANZRrnP6DKanx0YxdHAChxFq3rMsSaoziu71hMU810ZXgBvfa7KA30D+GzGBrnT2WsqYautDOq+upc+g7GNQYOl6k9/EeFLa6G/pfvkNnHcw/PHiC/dCHIfqJdjzRgcEflr12Ek9h3S33vxu+KdVf9pM3nXsKW6ITYdBOvwmLHAxCRvi4fnchmSsxlIloemDYb74xUpW534S7rsdnQovYa3/MqOk1QOdfmiOK1P/DVuCMI7xQjKZYG4Vr205Q5r86ed8aJsEvScnBzKSQCrvlsGOz5WK5im/mq6vV/GbAvqlvjzGqFH0ZkgfX2Y0Ajzjbs3gDpLZRHwlEMCzFrxUp4Wt1m4cf+8HmYltvNJXdZiwUyYXcYf+s6eAt8H4CN4yOsxMIAAA=', 'imet/utils.py': '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', 'imet/main.py': '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', 'setup.py': 'H4sIADnwoVwC/0srys9VKE4tKS0oyc/PKVbIzC3ILyqBiHBxgSkNLgUgyEvMTbVVz8xNLVHXAQsUJCZnJ6anFttGQ0Rjdbg0uQA9nF1NTwAAAA=='

# for path, encoded in file_data.items():
#     print(path)
#     path = Path(path)
#     path.parent.mkdir(exist_ok=True)
#     path.write_bytes(gzip.decompress(base64.b64decode(encoded)))

pythonVersion = 'python3'
def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:./kaggle/working && ' + command)


# run(pythonVersion+' setup.py develop --install-dir ./kaggle/working')
run(pythonVersion+' -m imet.make_folds --n-folds 40')
run(pythonVersion+' -m imet.main train model_3 --n-epochs 17 --model resnet101 --batch-size 32 --fold 0 --patience 3')
run(pythonVersion+' -m imet.main predict_test model_3 --model resnet101 --batch-size 32 --fold 0 --patience 3')
run(pythonVersion+' -m imet.make_submission model_3/test.h5 submission3.csv')