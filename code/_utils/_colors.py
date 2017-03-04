# -*- coding: UTF-8 -*-

_color_red_list = [
	{'percentage':'00%','color':'#ffffff'},
	{'percentage':'05%','color':'#ffe5e5'},
	{'percentage':'10%','color':'#ffcccc'},
	{'percentage':'15%','color':'#ffb3b3'},
	{'percentage':'20%','color':'#ff9999'},
	{'percentage':'25%','color':'#ff8080'},
	{'percentage':'30%','color':'#ff6666'},
	{'percentage':'35%','color':'#ff4d4d'},
	{'percentage':'40%','color':'#ff3333'},
	{'percentage':'45%','color':'#ff1a1a'},
	{'percentage':'50%','color':'#ff0000'},
	{'percentage':'55%','color':'#e60000'},
	{'percentage':'60%','color':'#cc0000'},
	{'percentage':'65%','color':'#b30000'},
	{'percentage':'70%','color':'#990000'},
	{'percentage':'75%','color':'#800000'},
	{'percentage':'80%','color':'#660000'},
	{'percentage':'85%','color':'#4d0000'},
	{'percentage':'90%','color':'#330000'},
	{'percentage':'95%','color':'#1a0000'}
]

_color_orange_list = [
	{'percentage':'00%','color':'#ffffff'},
	{'percentage':'05%','color':'#fff0e5'},
	{'percentage':'10%','color':'#ffe0cc'},
	{'percentage':'15%','color':'#ffd1b3'},
	{'percentage':'20%','color':'#ffc299'},
	{'percentage':'25%','color':'#ffb380'},
	{'percentage':'30%','color':'#ffa366'},
	{'percentage':'35%','color':'#ff944d'},
	{'percentage':'40%','color':'#ff8533'},
	{'percentage':'45%','color':'#ff751a'},
	{'percentage':'50%','color':'#ff6600'},
	{'percentage':'55%','color':'#e65c00'},
	{'percentage':'60%','color':'#cc5200'},
	{'percentage':'65%','color':'#b34700'},
	{'percentage':'70%','color':'#993d00'},
	{'percentage':'75%','color':'#803300'},
	{'percentage':'80%','color':'#662900'},
	{'percentage':'85%','color':'#4d1f00'},
	{'percentage':'90%','color':'#331400'},
	{'percentage':'95%','color':'#1a0a00'}
]

_color_yellow_list = [
	{'percentage':'00%','color':'#ffffff'},
	{'percentage':'05%','color':'#ffffe6'},
	{'percentage':'10%','color':'#ffffcc'},
	{'percentage':'15%','color':'#ffffb3'},
	{'percentage':'20%','color':'#ffff99'},
	{'percentage':'25%','color':'#ffff80'},
	{'percentage':'30%','color':'#ffff66'},
	{'percentage':'35%','color':'#ffff4d'},
	{'percentage':'40%','color':'#ffff33'},
	{'percentage':'45%','color':'#ffff1a'},
	{'percentage':'50%','color':'#ffff00'},
	{'percentage':'55%','color':'#e6e600'},
	{'percentage':'60%','color':'#cccc00'},
	{'percentage':'65%','color':'#b3b300'},
	{'percentage':'70%','color':'#999900'},
	{'percentage':'75%','color':'#808000'},
	{'percentage':'80%','color':'#666600'},
	{'percentage':'85%','color':'#4d4d00'},
	{'percentage':'90%','color':'#333300'},
	{'percentage':'95%','color':'#1a1a00'}
]

_color_green_list = [
	{'percentage':'00%','color':'#ffffff'},
	{'percentage':'05%','color':'#e6ffe6'},
	{'percentage':'10%','color':'#ccffcc'},
	{'percentage':'15%','color':'#b3ffb3'},
	{'percentage':'20%','color':'#99ff99'},
	{'percentage':'25%','color':'#80ff80'},
	{'percentage':'30%','color':'#66ff66'},
	{'percentage':'35%','color':'#4dff4d'},
	{'percentage':'40%','color':'#33ff33'},
	{'percentage':'45%','color':'#1aff1a'},
	{'percentage':'50%','color':'#00ff00'},
	{'percentage':'55%','color':'#00e600'},
	{'percentage':'60%','color':'#00cc00'},
	{'percentage':'65%','color':'#00b300'},
	{'percentage':'70%','color':'#009900'},
	{'percentage':'75%','color':'#008000'},
	{'percentage':'80%','color':'#006600'},
	{'percentage':'85%','color':'#004d00'},
	{'percentage':'90%','color':'#003300'},
	{'percentage':'95%','color':'#001a00'}
]

def get_color(_color_list, _min, _max, _value):
	return _color_list[int(.99*(_value-_min)/(_max-_min)*len(_color_list))]['color']
