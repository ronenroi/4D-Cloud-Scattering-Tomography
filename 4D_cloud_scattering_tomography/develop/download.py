from requests import Session
session = Session()
session.auth = ('roironen', 'rR7585779')
_redirect = session.get('https://asdc.larc.nasa.gov/data/AirMSPI/PODEX/ER2_GRP_ELLIPSOID/V005/2013/01/AirMSPI_ER2_GRP_ELLIPSOID_20130114_210543Z_CA-SanDiegoCounty_666F_F01_V005.hdf')
_response = session.get(_redirect.url)
with open('roi', 'wb') as file:
    file.write(_response._content)
