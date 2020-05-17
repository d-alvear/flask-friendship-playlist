//  <script type="text/javascript">

  function autoFill(songCategories) {
    const prepopulatedValues = {
      popVsRock: ["Bad Romance, Lady Gaga; Fantasy, Mariah Carey; Cry Me a River, Justin Timberlake","Smells Like Teen Spirit, Nirvana; Island In the Sun, Weezer; Never Let You Go, Third Eye Blind"],
      rockVsHiphop: ["The Middle, Jimmy Eat World; Mr. Brightside, The Killers; Californication, Red Hot Chili Peppers","Hey Ya, OutKast; In My Feelings, Drake; Waterfalls, TLC"],
      mixedGenres: ["Call Me Maybe, Carly Rae Jepsen; Dancing Queen, ABBA; All The Small Things, blink-182","Electric Feel, MGMT; Got To Give It Up, Marvin Gaye; Hotel California, Eagles"]
    }
    document.getElementsByName('query_a')[0].value = prepopulatedValues[songCategories][0];
    document.getElementsByName('query_b')[0].value = prepopulatedValues[songCategories][1];
 
    // var radioElements = document.getElementsByName("input3");

    // for (var i=0; i<radioElements.length; i++) {
    //   if (radioElements[i].getAttribute('value') == 'Radio3') {
    //     radioElements[i].checked = true;
    //   }
    // }
  }
// </script>

  