// Living Skin: the preset carousel. Chips filter, arrows scroll, a card
// writes "name|nonce" into the hidden textbox the Python side listens to.
(function () {
    const P = "ls";

    function app() {
        return typeof gradioApp === "function" ? gradioApp() : document;
    }

    function pick(car, card) {
        car.querySelectorAll("." + P + "-card." + P + "-on").forEach((c) => c.classList.remove(P + "-on"));
        card.classList.add(P + "-on");
        const box = app().querySelector("#" + car.dataset.target + " textarea, #" + car.dataset.target + " input");
        if (!box) return;
        box.value = card.dataset.name + "|" + Date.now();
        if (typeof updateInput === "function") updateInput(box);
        else box.dispatchEvent(new Event("input", {bubbles: true}));
    }

    function filter(car, chip) {
        car.querySelectorAll("." + P + "-chip").forEach((c) => c.classList.toggle(P + "-on", c === chip));
        const cat = chip.dataset.cat;
        car.querySelectorAll("." + P + "-card").forEach((c) => {
            c.hidden = cat !== "*" && c.dataset.cat !== cat;
        });
        const track = car.querySelector("." + P + "-track");
        if (track) track.scrollLeft = 0;
    }

    document.addEventListener("click", function (e) {
        const t = e.composedPath ? e.composedPath()[0] : e.target;
        if (!(t instanceof Element)) return;
        const car = t.closest("." + P + "-car");
        if (car) {
            const card = t.closest("." + P + "-card");
            const chip = t.closest("." + P + "-chip");
            const nav = t.closest("." + P + "-nav");
            if (card) pick(car, card);
            else if (chip) filter(car, chip);
            else if (nav) {
                const track = car.querySelector("." + P + "-track");
                track.scrollBy({left: Number(nav.dataset.dir) * track.clientWidth * 0.8, behavior: "smooth"});
            }
            return;
        }
        // Reset: no preset is picked any more.
        const reset = t.closest("[id^='" + P + "_reset_']");
        if (reset) {
            const tab = reset.id.slice((P + "_reset_").length);
            app().querySelectorAll("." + P + "-car[data-tab='" + tab + "'] ." + P + "-card." + P + "-on")
                .forEach((c) => c.classList.remove(P + "-on"));
        }
    });
})();
