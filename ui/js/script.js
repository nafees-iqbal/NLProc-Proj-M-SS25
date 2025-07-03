function handleMenuOpen() {
    document
      .getElementById("sidenav")
      .setAttribute("style", "display:block;left:0;top:0;transition: 0.5s;");
  }
  
  function handleMenuClose() {
    document
      .getElementById("sidenav")
      .setAttribute("style", "left:-400px;transition:0.5s;");
  }
  