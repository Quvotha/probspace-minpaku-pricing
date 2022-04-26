import re
import unicodedata


def normalize(s: str) -> str:
    return unicodedata.normalize('NFKC', s).lower().strip()


def remove_linefeed(name: str) -> str:
    return name.replace('\n', ' ')


def replace_simbols(name: str) -> str:
    simbols = '\~\\▼⭐○◇❉✳♕◎★●#☆♪□♬✤🎵♡✿【】,\"\'/();:：；#&◆≪≫,@!\*()+·$%[]➡︎・\|※、。丨_「」'
    for s in simbols:
        name = name.replace(s, ' ')
    name = name.replace('->', ' ')
    return name


def normalize_male_female_only(name: str) -> str:
    # female/male woman/man を見分けられるようにしないとおかしな結果になる
    patterns1 = '^(?!(fe|wo))(man only|maleonly|men only|male only|MenOnly|ManOnly|Men\'s-only)'
    name = re.sub(patterns1, '__maleonly', name)
    patterns2 = '(female only|ladies only|lady only|woman only|women only|ladiesonly|girls only|women\'s-only)'
    name = re.sub(patterns2, 'femaleonly', name)
    name = name.replace('femaleonly限女生', 'femaleonly 限女生')
    return name


def normalize_wifi(name: str) -> str:
    name = name.replace('wi-fi', 'wifi')
    name = name.replace('freewifi', ' free wifi')
    name = name.replace('free-wifi', ' free wifi')
    name = name.replace('poketwifi42m2', 'pocket wifi 42m2')

    # wifi 以外も直してる
    name = name.replace('ikebukuro.57m2.wifi.parking', 'ikebukuro 57m2 wifi parking')
    name = name.replace('3.near the ikebukuro.57m2.wifi.parking', '3 near the ikebukuro 57m2 wifi parking')
    name = name.replace('306wifi18', '306 wifi 18')
    name = name.replace('401wifi18', '401 wifi 18')
    name = name.replace('406wifi18', '406 wifi 18')
    name = name.replace('max6person jr 上野 浅草 秋葉原 3double bed proketwifi.601',
                        'max 6 person jr 上野 浅草 秋葉原 3 double bed pocket wifi 601')
    name = name.replace('narita.42min direct  aotosta.6min.wifi.english 中文',
                        'narita 42min direct aotosta 6min wifi english 中文')
    name = name.replace('maxwifi付き', 'max wifi 付き')

    name = name.replace('無料wifi', '無料 wifi')
    name = name.replace('高速wifi', '高速 wifi')
    name = name.replace('光wifi', '光 wifi')
    name = name.replace('免費wifi', '免費 wifi')
    name = name.replace('提供wifi付き', '提供 wifi 付き')
    name = name.replace('携帯wifiフリー', '携帯 wifi フリー')
    name = name.replace('wifi\'', 'wifi')

    name = name.replace('wifi無料', 'wifi 無料')
    name = name.replace('wifi完備', 'wifi 完備')
    name = name.replace('wifi有り', 'wifi 有り')
    name = name.replace('pocketwifi', 'pocket wifi')
    name = name.replace('pocket-wifi', 'pocket wifi')
    name = name.replace('bedswifi', 'beds wifi')
    name = name.replace('homewifi', 'home wifi')
    name = name.replace('.wifi ', '. wifi ')
    name = name.replace('ポケットwifi付', 'ポケット wifi 付')

    return name


def normalize_new_open(name: str) -> str:
    name = name.replace('new opened', 'new-open')
    name = name.replace('-new open-', 'new-open')
    name = name.replace('new open', 'new-open')
    name = name.replace('newly opened', 'new-open')
    name = name.replace('201.newopen', '201 new-open')
    match = re.search('[0-9]{3}newopen', name)
    if match:
        start, end = match.span()
        name = name[start:3] + ' ' + 'new-open' + name[end:]
    return name


def normalize_station_name(name: str) -> str:
    name = name.replace('sibuya', 'shibuya')
    name = name.replace('sinjuku', 'shinjuku')
    name = name.replace('shinzyuku', 'shinjuku')
    name = name.replace('sinagawa', 'shinagawa')
    name = name.replace('ryougoku', 'ryogoku')

    # 秋葉原
    name = name.replace('043akihabara-cozy', '043 akihabara cozy')
    name = name.replace('akihabarakiba', 'akihabara kiba')
    if re.match('^akibahara[^ ]', name):
        name = name[:9] + ' ' + name[9:]
    name = name.replace('akibahara', 'akihabara')

    # 他にもいろいろ修正
    name = name.replace('160m2meguro river villa 3bedroom in tokyo',
                        '160m to meguro river villa 3 bedroom in tokyo')

    # 駅？
    name = name.replace('sky tree', 'skytree')
    name = name.replace('tokyoskytree', 'tokyo skytree')
    return name


def remove_extra_space(name: str) -> str:
    name = name.replace('　', ' ')
    name = re.sub(' {2,}', ' ', name).strip()
    return name


def cleanse_name(name: str) -> str:
    funcs = (
        normalize,
        remove_linefeed,
        replace_simbols,
        normalize_male_female_only,
        normalize_wifi,
        normalize_new_open,
        normalize_station_name,
        remove_extra_space
    )
    for f in funcs:
        name = f(name)
    return name
