(function () {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const dropzoneEmpty = document.getElementById("dropzoneEmpty");
  const previewWrap = document.getElementById("previewWrap");
  const previewImg = document.getElementById("previewImg");
  const scanLine = document.getElementById("scanLine");
  const fileMeta = document.getElementById("fileMeta");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const resetBtn = document.getElementById("resetBtn");
  const errorBox = document.getElementById("errorBox");

  const idleState = document.getElementById("idleState");
  const loadingState = document.getElementById("loadingState");
  const resultState = document.getElementById("resultState");
  const loadingLog = document.getElementById("loadingLog");

  const gaugeFill = document.getElementById("gaugeFill");
  const scoreNumber = document.getElementById("scoreNumber");
  const riskLevel = document.getElementById("riskLevel");
  const riskSummary = document.getElementById("riskSummary");
  const indicatorsEl = document.getElementById("indicators");
  const explanationList = document.getElementById("explanationList");

  // Listing signals panel
  const signalsPanel = document.querySelector(".signals-panel");
  const signalsToggle = document.getElementById("signalsToggle");
  const priceInput = document.getElementById("priceInput");
  const expectedPriceInput = document.getElementById("expectedPriceInput");
  const avgRatingInput = document.getElementById("avgRatingInput");
  const numRatingsInput = document.getElementById("numRatingsInput");
  const sellerMonthsInput = document.getElementById("sellerMonthsInput");
  const reviewTextInput = document.getElementById("reviewTextInput");
  const reviewsCsvInput = document.getElementById("reviewsCsvInput");
  const buyerImagesInput = document.getElementById("buyerImagesInput");
  const buyerImagesPreview = document.getElementById("buyerImagesPreview");
  const signalsSummary = document.getElementById("signalsSummary");

  // Quick demo panel
  const demoSelect = document.getElementById("demoSelect");
  const demoLoadBtn = document.getElementById("demoLoadBtn");
  const demoReference = document.getElementById("demoReference");

  let demoProducts = [];
  let currentReviewTimestamps = [];
  let currentGroundTruth = null;
  let currentBuyerFiles = []; // File objects, from manual upload OR demo auto-load

  const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 70; // r=70

  const INDICATOR_LABELS = {
    visual_consistency: "Visual Consistency",
    image_quality: "Image Quality",
    texture: "Texture Analysis",
    compression: "Compression Artifacts",
    ai_ml_prediction: "AI / ML Prediction",
    reviews: "Review Authenticity",
    ratings: "Rating Pattern",
    price: "Price Comparison",
    velocity: "Review Timing",
    seller: "Seller Tenure",
    image_match: "Buyer Photo Match",
  };

  const INDICATOR_ORDER = [
    "visual_consistency",
    "image_quality",
    "texture",
    "compression",
    "ai_ml_prediction",
    "reviews",
    "ratings",
    "price",
    "velocity",
    "seller",
    "image_match",
  ];

  const LOG_MESSAGES = [
    "Reading pixel data…",
    "Measuring sharpness & noise…",
    "Scanning edge structures…",
    "Sampling texture patches…",
    "Checking color distribution…",
    "Estimating compression artifacts…",
    "Running local ML heuristics…",
    "Cross-checking listing signals…",
    "Compiling risk report…",
  ];

  let selectedFile = null;

  // ---------- Quick Demo ----------
  async function loadDemoCatalog() {
    try {
      const res = await fetch("/demo-products");
      demoProducts = await res.json();
      demoProducts.forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = p.title;
        demoSelect.appendChild(opt);
      });
    } catch (err) {
      // Demo catalog is optional - fail silently, manual upload still works.
    }
  }
  loadDemoCatalog();

  demoSelect.addEventListener("change", () => {
    demoLoadBtn.disabled = !demoSelect.value;
  });

  demoLoadBtn.addEventListener("click", async () => {
    const product = demoProducts.find((p) => p.id === demoSelect.value);
    if (!product) return;

    hideError();
    demoLoadBtn.disabled = true;
    demoLoadBtn.textContent = "Loading…";

    try {
      const imgRes = await fetch(product.image);
      const blob = await imgRes.blob();
      const file = new File([blob], product.image.split("/").pop(), { type: blob.type || "image/jpeg" });

      handleFile(file);

      // Fill listing signal fields
      priceInput.value = product.price ?? "";
      expectedPriceInput.value = product.market_price ?? "";
      avgRatingInput.value = product.avg_rating ?? "";
      numRatingsInput.value = product.num_ratings ?? "";
      sellerMonthsInput.value = product.seller_months_active ?? "";
      reviewTextInput.value = (product.sample_reviews || []).join("\n");
      reviewsCsvInput.value = "";

      currentReviewTimestamps = product.review_timestamps || [];
      currentGroundTruth = product.ground_truth || null;

      // Fetch any buyer-uploaded review images bundled with this demo product
      currentBuyerFiles = [];
      if (product.buyer_images && product.buyer_images.length) {
        const fetched = await Promise.all(
          product.buyer_images.map(async (url) => {
            try {
              const r = await fetch(url);
              const b = await r.blob();
              return new File([b], url.split("/").pop(), { type: b.type || "image/jpeg" });
            } catch (e) {
              return null;
            }
          })
        );
        currentBuyerFiles = fetched.filter(Boolean);
      }
      renderBuyerThumbs();

      signalsPanel.classList.add("expanded");
      updateSignalChips();
    } catch (err) {
      showError("Could not load demo product data.");
    } finally {
      demoLoadBtn.disabled = false;
      demoLoadBtn.textContent = "Load";
    }
  });

  // ---------- Listing signals toggle ----------
  signalsToggle.addEventListener("click", () => {
    signalsPanel.classList.toggle("expanded");
  });

  function renderBuyerThumbs() {
    buyerImagesPreview.innerHTML = "";
    currentBuyerFiles.forEach((file) => {
      const thumb = document.createElement("div");
      thumb.className = "buyer-thumb";
      const img = document.createElement("img");
      const reader = new FileReader();
      reader.onload = (e) => (img.src = e.target.result);
      reader.readAsDataURL(file);
      thumb.appendChild(img);
      buyerImagesPreview.appendChild(thumb);
    });
  }

  buyerImagesInput.addEventListener("change", () => {
    currentBuyerFiles = Array.from(buyerImagesInput.files || []);
    currentGroundTruth = null; // manual edit invalidates the demo reference
    renderBuyerThumbs();
    updateSignalChips();
  });

  function updateSignalChips() {
    // remove any dynamic chips except the base "Image" one
    signalsSummary.querySelectorAll(".signal-chip.dynamic").forEach((el) => el.remove());

    const chips = [];
    if (reviewTextInput.value.trim() || (reviewsCsvInput.files && reviewsCsvInput.files.length)) {
      chips.push("Reviews");
    }
    if (avgRatingInput.value && numRatingsInput.value) {
      chips.push("Ratings");
    }
    if (priceInput.value && expectedPriceInput.value) {
      chips.push("Price");
    }
    if (sellerMonthsInput.value) {
      chips.push("Seller Tenure");
    }
    if (currentReviewTimestamps && currentReviewTimestamps.length >= 3) {
      chips.push("Review Timing");
    }
    if (currentBuyerFiles && currentBuyerFiles.length) {
      chips.push("Buyer Photos");
    }

    chips.forEach((label) => {
      const span = document.createElement("span");
      span.className = "signal-chip dynamic";
      span.textContent = label;
      signalsSummary.appendChild(span);
    });
  }

  [priceInput, expectedPriceInput, avgRatingInput, numRatingsInput, sellerMonthsInput, reviewTextInput, reviewsCsvInput].forEach(
    (el) =>
      el.addEventListener("input", () => {
        currentGroundTruth = null; // fields were hand-edited, demo reference no longer applies
        updateSignalChips();
      })
  );

  // ---------- Upload handling ----------
  dropzone.addEventListener("click", () => {
    if (!previewWrap.classList.contains("hidden")) return;
    fileInput.click();
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length) handleFile(files[0]);
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length) handleFile(e.target.files[0]);
  });

  function handleFile(file) {
    const validTypes = ["image/png", "image/jpeg", "image/webp", "image/bmp"];
    if (!validTypes.includes(file.type)) {
      showError("Unsupported file type. Please upload a JPG, PNG, WEBP or BMP image.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showError("File too large. Max size is 10MB.");
      return;
    }

    hideError();
    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      dropzoneEmpty.classList.add("hidden");
      previewWrap.classList.remove("hidden");
      resetBtn.classList.remove("hidden");
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);

    fileMeta.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  }

  resetBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = "";
    previewImg.src = "";
    dropzoneEmpty.classList.remove("hidden");
    previewWrap.classList.add("hidden");
    resetBtn.classList.add("hidden");
    analyzeBtn.disabled = true;
    fileMeta.textContent = "";
    hideError();
    showIdle();

    // clear listing signals + demo state
    priceInput.value = "";
    expectedPriceInput.value = "";
    avgRatingInput.value = "";
    numRatingsInput.value = "";
    sellerMonthsInput.value = "";
    reviewTextInput.value = "";
    reviewsCsvInput.value = "";
    buyerImagesInput.value = "";
    currentBuyerFiles = [];
    renderBuyerThumbs();
    demoSelect.value = "";
    demoLoadBtn.disabled = true;
    currentReviewTimestamps = [];
    currentGroundTruth = null;
    updateSignalChips();
  });

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.remove("hidden");
  }
  function hideError() {
    errorBox.classList.add("hidden");
  }

  // ---------- Analyze ----------
  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    hideError();
    showLoading();

    const formData = new FormData();
    formData.append("image", selectedFile);

    if (priceInput.value) formData.append("price", priceInput.value);
    if (expectedPriceInput.value) formData.append("expected_price", expectedPriceInput.value);
    if (avgRatingInput.value) formData.append("avg_rating", avgRatingInput.value);
    if (numRatingsInput.value) formData.append("num_ratings", numRatingsInput.value);
    if (sellerMonthsInput.value) formData.append("seller_months_active", sellerMonthsInput.value);
    if (reviewTextInput.value.trim()) formData.append("review_text", reviewTextInput.value.trim());
    if (currentReviewTimestamps && currentReviewTimestamps.length) {
      formData.append("review_timestamps", JSON.stringify(currentReviewTimestamps));
    }
    if (reviewsCsvInput.files && reviewsCsvInput.files.length) {
      formData.append("reviews_csv", reviewsCsvInput.files[0]);
    }
    currentBuyerFiles.forEach((file) => formData.append("buyer_images", file));

    try {
      const res = await fetch("/analyze", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Analysis failed.");
      }

      await sleep(400);
      renderResult(data);
    } catch (err) {
      showIdle();
      showError(err.message || "Something went wrong while analyzing the image.");
    }
  });

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // ---------- State switching ----------
  function showIdle() {
    idleState.classList.remove("hidden");
    loadingState.classList.add("hidden");
    resultState.classList.add("hidden");
    scanLine.classList.add("hidden");
  }

  let logInterval = null;

  function showLoading() {
    idleState.classList.add("hidden");
    loadingState.classList.remove("hidden");
    resultState.classList.add("hidden");
    scanLine.classList.remove("hidden");
    analyzeBtn.disabled = true;

    let i = 0;
    loadingLog.textContent = LOG_MESSAGES[0];
    clearInterval(logInterval);
    logInterval = setInterval(() => {
      i = (i + 1) % LOG_MESSAGES.length;
      loadingLog.textContent = LOG_MESSAGES[i];
    }, 450);
  }

  function showResult() {
    clearInterval(logInterval);
    idleState.classList.add("hidden");
    loadingState.classList.add("hidden");
    resultState.classList.remove("hidden");
    scanLine.classList.add("hidden");
    analyzeBtn.disabled = false;
  }

  // ---------- Rendering ----------
  function levelColor(level) {
    if (level === "HIGH") return "var(--high)";
    if (level === "MEDIUM") return "var(--medium)";
    return "var(--low)";
  }

  function scoreColor(value) {
    if (value >= 70) return "var(--high)";
    if (value >= 40) return "var(--medium)";
    return "var(--low)";
  }

  function summaryFor(level, score, signalsUsed) {
    const signalsNote =
      signalsUsed.length > 1
        ? ` Based on ${signalsUsed.length} combined signals: ${signalsUsed.join(", ")}.`
        : " Based on image analysis only — add listing details above for a more grounded score.";

    if (level === "HIGH") {
      return `This listing shows multiple strong indicators (score ${score}/100) commonly associated with counterfeit or misleading listings.${signalsNote}`;
    }
    if (level === "MEDIUM") {
      return `This listing shows some indicators (score ${score}/100) worth a closer look.${signalsNote}`;
    }
    return `This listing shows few or no indicators (score ${score}/100) typically associated with counterfeit listings.${signalsNote}`;
  }

  function buildIndicatorValues(data) {
    const values = { ...data.sub_scores };
    if (data.review_analysis) values.reviews = data.review_analysis.score;
    if (data.rating_analysis) values.ratings = data.rating_analysis.score;
    if (data.price_analysis) values.price = data.price_analysis.score;
    if (data.velocity_analysis) values.velocity = data.velocity_analysis.score;
    if (data.seller_analysis) values.seller = data.seller_analysis.score;
    if (data.image_match_analysis) values.image_match = data.image_match_analysis.score;
    return values;
  }

  function renderResult(data) {
    showResult();

    // Gauge
    const pct = Math.max(0, Math.min(100, data.score)) / 100;
    const offset = GAUGE_CIRCUMFERENCE * (1 - pct);
    gaugeFill.style.stroke = levelColor(data.level);
    gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        gaugeFill.style.strokeDashoffset = offset;
      });
    });

    scoreNumber.textContent = data.score;
    scoreNumber.style.color = levelColor(data.level);

    riskLevel.textContent = data.level;
    riskLevel.className = "risk-level " + data.level.toLowerCase();

    riskSummary.textContent = summaryFor(data.level, data.score, data.signals_used || ["Image Analysis"]);

    if (currentGroundTruth) {
      const labelMap = {
        genuine: "Demo dataset reference: labeled GENUINE",
        fake: "Demo dataset reference: labeled FAKE",
        unlabeled: "Demo dataset reference: no ground truth (relies on AI/image + available signals)",
      };
      demoReference.textContent = labelMap[currentGroundTruth] || "";
      demoReference.className = "demo-reference " + currentGroundTruth;
      demoReference.classList.remove("hidden");
    } else {
      demoReference.classList.add("hidden");
    }

    // Indicators (image sub-scores + any optional signals provided)
    const values = buildIndicatorValues(data);
    indicatorsEl.innerHTML = "";
    INDICATOR_ORDER.forEach((key) => {
      if (!(key in values)) return;
      const value = values[key];
      const card = document.createElement("div");
      card.className = "indicator-card";
      card.innerHTML = `
        <div class="indicator-head">
          <span class="indicator-name">${INDICATOR_LABELS[key]}</span>
          <span class="indicator-value" style="color:${scoreColor(value)}">${value}</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="background:${scoreColor(value)}"></div>
        </div>
      `;
      indicatorsEl.appendChild(card);
      const fill = card.querySelector(".bar-fill");
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          fill.style.width = value + "%";
        });
      });
    });

    // Explanation
    explanationList.innerHTML = "";
    data.explanation.forEach((reason) => {
      const li = document.createElement("li");
      li.textContent = reason;
      explanationList.appendChild(li);
    });
  }
})();
