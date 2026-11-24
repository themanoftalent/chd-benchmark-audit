"""Retrieve UCI archives locally; copy a compilation obtained from its provider."""
import argparse, hashlib, json, shutil, urllib.request, zipfile
from pathlib import Path

URLS={
 'heart-disease.zip':'https://archive.ics.uci.edu/static/public/45/heart+disease.zip',
 'statlog-heart.zip':'https://archive.ics.uci.edu/static/public/145/statlog+heart.zip',
}
FILES=['processed.cleveland.data','processed.hungarian.data','processed.switzerland.data','processed.va.data','heart.dat','heart_disease_uci.csv']

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def unpack(archive, dest, wanted):
    with zipfile.ZipFile(archive) as z:
        for name in wanted:
            hits=[n for n in z.namelist() if Path(n).name==name and not n.endswith('/')]
            if len(hits)!=1: raise ValueError(f'{archive}: expected one {name}, got {len(hits)}')
            # Write only explicitly named members, never arbitrary archive paths.
            (dest/name).write_bytes(z.read(hits[0]))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('data'));p.add_argument('--compilation',type=Path,help='CSV downloaded from Kaggle');p.add_argument('--uci-local',type=Path,help='Directory containing both downloaded UCI ZIP archives');p.add_argument('--expected-hashes',type=Path,help='JSON mapping of trusted filenames to SHA-256 hashes');p.add_argument('--verify-only',action='store_true');a=p.parse_args();a.data.mkdir(parents=True,exist_ok=True)
    if not a.verify_only:
        for name,url in URLS.items():
            dst=a.data/name
            if a.uci_local: shutil.copyfile(a.uci_local/name,dst)
            elif not dst.exists():
                tmp=dst.with_suffix('.part')
                try:
                    with urllib.request.urlopen(url,timeout=90) as response,tmp.open('wb') as f: shutil.copyfileobj(response,f)
                    tmp.replace(dst)
                except Exception: tmp.unlink(missing_ok=True);raise
            unpack(dst,a.data,FILES[:4] if name=='heart-disease.zip' else ['heart.dat'])
        if a.compilation: shutil.copyfile(a.compilation,a.data/'heart_disease_uci.csv')
    absent=[f for f in FILES if not (a.data/f).is_file()]
    if absent: raise SystemExit('Missing inputs: '+', '.join(absent)+'; see data/README.md')
    hashes={f:digest(a.data/f) for f in FILES}
    if a.expected_hashes:
        expected=json.loads(a.expected_hashes.read_text()); mismatch=[f for f in FILES if hashes[f]!=expected.get(f)]
        if mismatch: raise SystemExit('SHA-256 mismatch: '+', '.join(mismatch))
        print('All input hashes match the supplied trusted manifest.')
    else: print('Input hashes calculated; no trusted original manifest supplied.')
    (a.data/'input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('Inputs ready:',a.data)
if __name__=='__main__': main()
