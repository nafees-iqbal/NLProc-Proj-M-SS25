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
async function sendQuestion(question) {
    const response = await fetch('http://127.0.0.1:8000/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question: question, task: "qa" })
    });

    const data = await response.json();
    return data.answer;
}

// Setup Enter key listener
document.querySelector("#chat-section input").addEventListener("keydown", async (e) => {
    if (e.key === "Enter") {
        const userInput = e.target.value.trim();
        if (!userInput) return;

        const messagesDiv = document.getElementById("messages");

        const userMsg = document.createElement("div");
        userMsg.className = "user-message message";
        userMsg.innerHTML = `<div class="message-div">
          <div class="message-profile-pic"><img src="img/user.png" height="30" width="30" /></div>
          <div class="message-content"><p>${userInput}</p></div>
        </div>`;
        messagesDiv.appendChild(userMsg);

        e.target.value = "";

        const botMsg = document.createElement("div");
        botMsg.className = "gpt-message message";
        botMsg.innerHTML = `<div class="message-div">
          <div class="message-profile-pic"><img src="img/chat-gpt.png" height="30" width="30" /></div>
          <div class="message-content"><p>Loading...</p></div>
        </div>`;
        messagesDiv.appendChild(botMsg);

        const answer = await sendQuestion(userInput);

        botMsg.querySelector(".message-content p").textContent = answer;

        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }
});



  