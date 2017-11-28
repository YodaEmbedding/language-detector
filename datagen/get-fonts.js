/*

Acquire fonts from https://fonts.google.com/

print('\n'.join(['https://fonts.google.com/?category=Sans+Serif&selection.family=' + '|'.join(x) for x in chunks(fontstr.split('|'), 16)]))

Sans serif:

https://fonts.google.com/?category=Sans+Serif&selection.family=ABeeZee|Abel|Aclonica|Acme|Actor|Advent+Pro|Aldrich|Alef|Alegreya+Sans|Alegreya+Sans+SC|Allerta|Allerta+Stencil|Amaranth|Amiko|Anaheim|Andika|Antic|Anton|Archivo|Archivo+Black|Archivo+Narrow|Arimo|Armata|Arsenal|Arya|Asap|Asap+Condensed|Assistant|Asul|Athiti|Average+Sans|Barlow|Barlow+Condensed|Barlow+Semi+Condensed|Basic|Belleza|BenchNine|Biryani|Bubbler+One|Cabin|Cabin+Condensed|Cagliostro|Cairo|Cambay|Candal|Cantarell|Cantora+One|Capriola|Carme|Carrois+Gothic|Carrois+Gothic+SC|Catamaran|Changa|Chathura|Chau+Philomene+One|Chivo|Coda+Caption:800|Convergence|Cuprum|Days+One|Denk+One|Dhurjati|Didact+Gothic|Doppio+One|Dorsa|Dosis|Duru+Sans|Economica|El+Messiri|Electrolize|Encode+Sans|Encode+Sans+Condensed|Encode+Sans+Expanded|Encode+Sans+Semi+Condensed|Encode+Sans+Semi+Expanded|Englebert|Exo|Exo+2|Federo|Fira+Sans|Fira+Sans+Condensed|Fira+Sans+Extra+Condensed|Fjalla+One|Francois+One|Fresca|GFS+Neohellenic|Gafata|Galdeano|Geo|Gidugu|Gudea|Hammersmith+One|Harmattan|Heebo|Hind|Hind+Guntur|Hind+Madurai|Hind+Siliguri|Hind+Vadodara|Homenaje|Imprima|Inder|Istok+Web|Jaldi|Jockey+One|Josefin+Sans|Julius+Sans+One|Jura|Kanit|Kantumruy|Karla|Khand|Khula|Kite+One|Krona+One|Lato|Lekton|Libre+Franklin|Mada|Magra|Mako|Mallanna|Mandali|Marmelad|Martel+Sans|Marvel|Maven+Pro|Meera+Inimai|Merriweather+Sans|Metrophobic|Michroma|Miriam+Libre|Mitr|Molengo|Monda|Montserrat|Montserrat+Alternates|Montserrat+Subrayada|Mouse+Memoirs|Mukta|Mukta+Mahee|Mukta+Malar|Mukta+Vaani|Muli|NTR|News+Cycle|Nobile|Noto+Sans|Numans|Nunito|Nunito+Sans|Open+Sans|Open+Sans+Condensed:300|Orbitron|Orienta|Oswald|Overpass|Oxygen|PT+Sans|PT+Sans+Caption|PT+Sans+Narrow|Padauk|Palanquin|Palanquin+Dark|Pathway+Gothic+One|Pattaya|Pavanam|Paytone+One|Philosopher|Play|Pontano+Sans|Poppins|Port+Lligat+Sans|Pragati+Narrow|Prompt|Proza+Libre|Puritan|Quantico|Quattrocento+Sans|Questrial|Quicksand|Rajdhani|Raleway|Ramabhadra|Rambla|Rationale|Reem+Kufi|Roboto|Roboto+Condensed|Ropa+Sans|Rosario|Rubik|Rubik+Mono+One|Ruda|Ruluko|Rum+Raisin|Russo+One|Saira|Saira+Condensed|Saira+Extra+Condensed|Saira+Semi+Condensed|Sansita|Sarala|Sarpanch|Scada|Secular+One|Seymour+One|Shanti|Share+Tech|Signika|Signika+Negative|Sintony|Six+Caps|Snippet|Source+Sans+Pro|Spinnaker|Strait|Syncopate|Tauri|Teko|Telex|Tenali+Ramakrishna|Tenor+Sans|Text+Me+One|Timmana|Titillium+Web|Ubuntu|Ubuntu+Condensed|Varela|Varela+Round|Viga|Voltaire|Wendy+One|Wire+One|Work+Sans|Yanone+Kaffeesatz|Yantramanav

*/

/*
function selectOnPage() {
	for (var button of document.getElementsByClassName('selection-toggle-button')) {
		var isSelectable = button.getAttribute('aria-label').startsWith('Select');

		if (isSelectable) {
			button.click();
		}
	}
}

(function loop(i) {
	setTimeout(function() {
		selectOnPage();
		window.scrollByLines(10)
		if (--i) loop(i);
	}, 500)
})(50);

var unselectButtons = document.getElementsByClassName('md-icon-button selection-family-chip-deselect md-button md-ink-ripple');
var btnDownload = document.getElementsByClassName('md-icon-button collection-drawer-download-button md-button md-ink-ripple')[0];

var counter = 16;
for (var button of unselectButtons) {
	if (counter == 16) {
		counter = 0;
		btnDownload.click();
	}

	button.click();
	counter++;
}*/


// Polyfill
// https://tc39.github.io/ecma262/#sec-array.prototype.includes
if (!Array.prototype.includes) {
  Object.defineProperty(Array.prototype, 'includes', {
    value: function(searchElement, fromIndex) {

      if (this == null) {
        throw new TypeError('"this" is null or not defined');
      }

      // 1. Let O be ? ToObject(this value).
      var o = Object(this);

      // 2. Let len be ? ToLength(? Get(O, "length")).
      var len = o.length >>> 0;

      // 3. If len is 0, return false.
      if (len === 0) {
        return false;
      }

      // 4. Let n be ? ToInteger(fromIndex).
      //    (If fromIndex is undefined, this step produces the value 0.)
      var n = fromIndex | 0;

      // 5. If n ≥ 0, then
      //  a. Let k be n.
      // 6. Else n < 0,
      //  a. Let k be len + n.
      //  b. If k < 0, let k be 0.
      var k = Math.max(n >= 0 ? n : len - Math.abs(n), 0);

      function sameValueZero(x, y) {
        return x === y || (typeof x === 'number' && typeof y === 'number' && isNaN(x) && isNaN(y));
      }

      // 7. Repeat, while k < len
      while (k < len) {
        // a. Let elementK be the result of ? Get(O, ! ToString(k)).
        // b. If SameValueZero(searchElement, elementK) is true, return true.
        if (sameValueZero(o[k], searchElement)) {
          return true;
        }
        // c. Increase k by 1. 
        k++;
      }

      // 8. Return false
      return false;
    }
  });
}


//var nameHistory = [];
var nameHistory = new Set([]);

function selectOnPage(count) {
	var selectableButtons = document.getElementsByClassName('selection-toggle-button');

	for (var button of selectableButtons) {
		if (count == 0) {
			break;
		}

		var name = button.parentElement.parentElement.children[1].children[0].innerText;

		if (nameHistory.has(name) === -1) {
			break;
		}

		nameHistory.add(name);

		if (button.getAttribute('aria-label').startsWith('Select')) {
			button.click();
			count--;
		}
	}

	return count;
}

function download() {
	var btnDownload = document.getElementsByClassName('md-icon-button collection-drawer-download-button md-button md-ink-ripple')[0];
	btnDownload.click();
}

function unselectAll() {
	var unselectButtons = document.getElementsByClassName('md-icon-button selection-family-chip-deselect md-button md-ink-ripple');

	for (var button of unselectButtons) {
		button.click();
	}
}

function selectTillDownloadable() {
	var remaining = selectOnPage(16);
	while(remaining != 0 && window.scrollY < window.scrollMaxY) {
		remaining = selectOnPage(remaining);
		window.scrollByLines(10);
	}
}

(function loop(i) {
	setTimeout(function() {
		unselectAll();
		selectTillDownloadable();
		//download();

		if (--i) {
			loop(i);
		}
	}, 5000)
})(20);


