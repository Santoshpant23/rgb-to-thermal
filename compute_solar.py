"""Compute per-image solar elevation/azimuth from timestamp + lat/long -> data_cache/solar.json.
The sun's position is the one physical signal that actually varies per image (captured across ~1h)
and physically drives surface heating (sun-facing vs shaded)."""
import json, math, os
import r2t_common as C
META=f"{C.BASE}/drone_and_weather_metadata.json"; SPLIT=f"{C.BASE}/code/train_test_split.json"
TZ_OFFSET=-4  # Ann Arbor EDT (Aug) -> UTC = local - (-4) = local + 4

def solar_pos(ts, lat, lng):
    # ts "YYYY:MM:DD HH:MM:SS" local; convert to UTC
    d,t=ts.split(" "); Y,Mo,D=[int(x) for x in d.split(":")]; h,mi,s=[int(x) for x in t.split(":")]
    # day of year
    import datetime
    N=datetime.date(Y,Mo,D).timetuple().tm_yday
    utc_h=h+mi/60+s/3600 - TZ_OFFSET     # local + 4 = UTC
    decl=math.radians(23.45*math.sin(math.radians(360/365*(284+N))))
    # local solar time approx: UTC hours + lng/15
    lst=utc_h + lng/15.0
    H=math.radians(15*(lst-12))
    la=math.radians(lat)
    sin_el=math.sin(la)*math.sin(decl)+math.cos(la)*math.cos(decl)*math.cos(H)
    el=math.degrees(math.asin(max(-1,min(1,sin_el))))
    az=math.degrees(math.atan2(math.sin(H), math.cos(H)*math.sin(la)-math.tan(decl)*math.cos(la)))
    az=(az+180)%360
    return el, az

def main():
    meta=json.load(open(META)); split=json.load(open(SPLIT))
    out={}; els=[]; azs=[]
    for simple,v in split.items():
        dji=v[0]; e=meta.get(dji)
        if not e or 'timestamp' not in e: continue
        try:
            el,az=solar_pos(e['timestamp'], e.get('lat',42.28), e.get('lng',-83.74))
            n=simple.replace('.JPG',''); out[n]=[round(el,3),round(az,3)]; els.append(el); azs.append(az)
        except Exception as ex: pass
    json.dump(out, open(f"{C.CACHE}/solar.json","w"))
    import numpy as np
    print(f"solar.json: {len(out)} entries. elev range [{min(els):.2f},{max(els):.2f}] (var {np.std(els):.2f}) "
          f"azim range [{min(azs):.2f},{max(azs):.2f}] (var {np.std(azs):.2f})")

if __name__=='__main__': main()
