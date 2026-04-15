import { useState, useMemo } from "react";

const data = {
  "Yahoo DSP": {
    color: "#6001D2",
    taxonomy: "IAB Content Taxonomy v2.x",
    channels: {
      "Standard Display": { status: "full", icon: "🖥️" },
      "Native Display": { status: "full", icon: "📰" },
      "Standard Video": { status: "full", icon: "🎬" },
      "Native Video": { status: "full", icon: "📹" },
      "Connected TV (CTV)": { status: "full", icon: "📺" },
      "Audio": { status: "full", icon: "🎧" },
      "DOOH": { status: "limited", icon: "🏙️" }
    }
  },
  "The Trade Desk": {
    color: "#00B140",
    taxonomy: "IAB v2.x + v3.0 (1,500+ categories)",
    channels: {
      "Display (Web)": { status: "full", icon: "🖥️" },
      "Display (In-App)": { status: "full", icon: "📱" },
      "Video (Web)": { status: "full", icon: "🎬" },
      "Video (In-App)": { status: "full", icon: "📹" },
      "Native": { status: "full", icon: "📰" },
      "Connected TV (CTV)": { status: "full", icon: "📺" },
      "Audio / Podcast": { status: "full", icon: "🎧" },
      "DOOH": { status: "limited", icon: "🏙️" }
    }
  },
  "DV360": {
    color: "#4285F4",
    taxonomy: "Google Proprietary (IAB-aligned)",
    channels: {
      "Display (Web)": { status: "full", icon: "🖥️", note: "Google Content Categories" },
      "Display (In-App)": { status: "full", icon: "📱", note: "App Categories" },
      "Video (Web)": { status: "full", icon: "🎬", note: "Google Content Categories" },
      "Connected TV (CTV)": { status: "genre", icon: "📺", note: "Genre Targeting (subset)" },
      "Audio": { status: "genre", icon: "🎧", note: "Genre Targeting (subset)" },
      "YouTube (TrueView)": { status: "topics", icon: "▶️", note: "Google Topics (2,000+)" }
    }
  }
};

const iabCategories = [
  { id: "IAB1", name: "Arts & Entertainment", subs: ["Books & Literature","Celebrity Fan/Gossip","Fine Art","Humor","Movies","Music","Television"] },
  { id: "IAB2", name: "Automotive", subs: ["Auto Parts","Auto Repair","Buying/Selling Cars","Car Culture","Certified Pre-Owned","Convertible","Coupe","Crossover","Diesel","Electric Vehicle","Hatchback","Hybrid","Luxury","Minivan","Motorcycles","Off-Road Vehicles","Performance Vehicles","Pickup","Road-Side Assistance","Sedan","Trucks & Accessories","Vintage Cars","Wagon"] },
  { id: "IAB3", name: "Business", subs: ["Advertising","Agriculture","Biotech/Biomedical","Business Software","Construction","Forestry","Government","Green Solutions","Human Resources","Logistics","Marketing","Metals"] },
  { id: "IAB4", name: "Careers", subs: ["Career Planning","College","Financial Aid","Job Fairs","Job Search","Resume Writing/Advice","Nursing","Scholarships","Telecommuting","U.S. Military","Career Advice"] },
  { id: "IAB5", name: "Education", subs: ["7-12 Education","Adult Education","Art History","College Administration","College Life","Distance Learning","English as a 2nd Language","Language Learning","Graduate School","Homeschooling","Homework/Study Tips","K-6 Educators","Private School","Special Education","Studying Business"] },
  { id: "IAB6", name: "Family & Parenting", subs: ["Adoption","Babies & Toddlers","Daycare/Pre School","Family Internet","Parenting – K-6 Kids","Parenting – Teens","Pregnancy","Special Needs Kids","Eldercare"] },
  { id: "IAB7", name: "Health & Fitness", subs: ["Exercise","ADD","AIDS/HIV","Allergies","Alternative Medicine","Arthritis","Asthma","Autism/PDD","Bipolar Disorder","Brain Tumor","Cancer","Cholesterol","Chronic Fatigue Syndrome","Chronic Pain","Cold & Flu","Deafness","Dental Care","Depression","Dermatology","Diabetes","Epilepsy","GERD/Acid Reflux","Headaches/Migraines","Heart Disease","Herbs for Health","Holistic Healing","IBS/Crohn's Disease","Incest/Abuse Support","Incontinence","Infertility","Men's Health","Nutrition","Orthopedics","Panic/Anxiety Disorders","Pediatrics","Physical Therapy","Psychology/Psychiatry","Senior Health","Sexuality","Sleep Disorders","Smoking Cessation","Substance Abuse","Thyroid Disease","Weight Loss","Women's Health"] },
  { id: "IAB8", name: "Food & Drink", subs: ["American Cuisine","Barbecues & Grilling","Cajun/Creole","Chinese Cuisine","Cocktails/Beer","Coffee/Tea","Cuisine-Specific","Desserts & Baking","Dining Out","Food Allergies","French Cuisine","Health/Low-Fat Cooking","Italian Cuisine","Japanese Cuisine","Mexican Cuisine","Vegan","Vegetarian","Wine"] },
  { id: "IAB9", name: "Hobbies & Interests", subs: ["Art/Technology","Arts & Crafts","Beadwork","Bird Watching","Board Games/Puzzles","Candle & Soap Making","Card Games","Chess","Cigars","Collecting","Comic Books","Drawing/Sketching","Freelance Writing","Genealogy","Getting Published","Guitar","Home Recording","Investors & Patents","Jewelry Making","Magic & Illusion","Needlework","Painting","Photography","Radio","Roleplaying Games","Sci-Fi & Fantasy","Scrapbooking","Screenwriting","Stamps & Coins","Video & Computer Games","Woodworking"] },
  { id: "IAB10", name: "Home & Garden", subs: ["Appliances","Entertaining","Environmental Safety","Gardening","Home Repair","Home Theater","Interior Decorating","Landscaping","Remodeling & Construction"] },
  { id: "IAB11", name: "Law, Gov & Politics", subs: ["Immigration","Legal Issues","U.S. Government Resources","Politics","Commentary"] },
  { id: "IAB12", name: "News", subs: ["International News","National News","Local News"] },
  { id: "IAB13", name: "Personal Finance", subs: ["Beginning Investing","Credit/Debt & Loans","Financial News","Financial Planning","Hedge Fund","Insurance","Investing","Mutual Funds","Options","Retirement Planning","Stocks","Tax Planning"] },
  { id: "IAB14", name: "Society", subs: ["Dating","Divorce Support","Gay Life","Marriage","Senior Living","Teens","Weddings","Ethnic Specific"] },
  { id: "IAB15", name: "Science", subs: ["Astrology","Biology","Chemistry","Geology","Paranormal Phenomena","Physics","Space/Astronomy","Geography","Botany","Weather"] },
  { id: "IAB16", name: "Pets", subs: ["Aquariums","Birds","Cats","Dogs","Large Animals","Reptiles","Veterinary Medicine"] },
  { id: "IAB17", name: "Sports", subs: ["Auto Racing","Baseball","Bicycling","Bodybuilding","Boxing","Canoeing/Kayaking","Cheerleading","Climbing","Cricket","Figure Skating","Fly Fishing","Football","Freshwater Fishing","Game & Fish","Golf","Horse Racing","Horses","Hunting/Shooting","Inline Skating","Martial Arts","Mountain Biking","NASCAR Racing","Olympics","Paintball","Power & Motorcycles","Pro Basketball","Pro Ice Hockey","Rodeo","Rugby","Running/Jogging","Sailing","Saltwater Fishing","Scuba Diving","Skateboarding","Skiing","Snowboarding","Surfing/Body-Boarding","Swimming","Table Tennis/Ping-Pong","Tennis","Volleyball","Walking","Waterski/Wakeboard","World Soccer"] },
  { id: "IAB18", name: "Style & Fashion", subs: ["Beauty","Body Art","Fashion","Jewelry","Clothing","Accessories"] },
  { id: "IAB19", name: "Technology & Computing", subs: ["3-D Graphics","Animation","Antivirus Software","C/C++","Cameras & Camcorders","Cell Phones","Computer Certification","Computer Networking","Computer Peripherals","Computer Reviews","Data Centers","Databases","Desktop Publishing","Desktop Video","Email","Graphics Software","Home Video/DVD","Internet Technology","Java","JavaScript","Mac Support","MP3/MIDI","Net Conferencing","Net for Beginners","Network Security","Palmtops/PDAs","PC Support","Portable Entertainment","Shareware/Freeware","Linux","Visual Basic","Web Design/HTML","Web Search","Windows"] },
  { id: "IAB20", name: "Travel", subs: ["Adventure Travel","Africa","Air Travel","Australia & New Zealand","Bed & Breakfasts","Budget Travel","Business Travel","By US Locale","Camping","Canada","Caribbean","Cruises","Eastern Europe","Europe","France","Greece","Honeymoons/Getaways","Hotels","Italy","Japan","Mexico & Central America","National Parks","South America","Spas","Theme Parks","Traveling with Kids","United Kingdom"] },
  { id: "IAB21", name: "Real Estate", subs: ["Apartments","Architects","Buying/Selling Homes"] },
  { id: "IAB22", name: "Shopping", subs: ["Contests & Freebies","Coupons","Comparison","Engines"] },
  { id: "IAB23", name: "Religion & Spirituality", subs: ["Alternative Religions","Atheism/Agnosticism","Buddhism","Catholicism","Christianity","Hinduism","Islam","Judaism","Latter-Day Saints","Pagan/Wiccan"] },
  { id: "IAB24", name: "Uncategorized", subs: [] },
  { id: "IAB25", name: "Non-Standard Content", subs: ["Unmoderated UGC","Extreme Graphic/Explicit Violence","Pornography","Profane Content","Hate Content","Under Construction","Incentivized"] },
  { id: "IAB26", name: "Illegal Content", subs: ["Illegal Content","Warez","Spyware/Malware","Copyright Infringement"] }
];

const dv360Categories = [
  "Arts & Entertainment","Autos & Vehicles","Beauty & Fitness","Books & Literature","Business & Industrial","Computers & Electronics","Finance","Food & Drink","Games","Health","Hobbies & Leisure","Home & Garden","Internet & Telecom","Jobs & Education","Law & Government","News","Online Communities","People & Society","Pets & Animals","Real Estate","Reference","Science","Sensitive Subjects","Shopping","Sports","Travel","World Localities"
];

const dv360Genres = [
  "Arts & Entertainment","Comedy","Drama","Education","Games","Lifestyle","Movies","Music & Audio","News & Politics","Reality","Science & Nature","Sports","TV & Video"
];

const sensitiveCategories = {
  "Yahoo DSP": ["IAB25 Non-Standard Content","IAB26 Illegal Content","Adult","Violence","Hate Speech","Drugs & Alcohol","Gambling","Profanity"],
  "The Trade Desk": ["Adult/Explicit Sexual Content","Arms & Ammunition","Crime & Harmful Acts","Death, Injury or Military Conflict","Online Piracy","Hate Speech & Acts of Aggression","Obscenity & Profanity","Illegal Drugs/Tobacco/Vaping/Alcohol","Spam or Harmful Content","Terrorism","Sensitive Social Issues"],
  "DV360": ["Adult","Derogatory","Downloads & Sharing","Weapons","Gambling","Violence","Suggestive","Profanity","Alcohol","Drugs","Tobacco","Politics","Religion","Tragedy","Transportation Accidents","Sensitive Social Issues","Shocking"]
};

const Badge = ({ type }) => {
  const styles = {
    full: { bg: "#DCFCE7", color: "#166534", text: "Full IAB" },
    limited: { bg: "#FEF9C3", color: "#854D0E", text: "Limited" },
    genre: { bg: "#DBEAFE", color: "#1E40AF", text: "Genre Only" },
    topics: { bg: "#F3E8FF", color: "#6B21A8", text: "Topics" }
  };
  const s = styles[type] || styles.full;
  return <span style={{ background: s.bg, color: s.color, padding: "2px 8px", borderRadius: 12, fontSize: 11, fontWeight: 600 }}>{s.text}</span>;
};

export default function App() {
  const [platform, setPlatform] = useState("Yahoo DSP");
  const [selectedChannel, setSelectedChannel] = useState(null);
  const [expandedCat, setExpandedCat] = useState(null);
  const [search, setSearch] = useState("");
  const [tab, setTab] = useState("categories");

  const p = data[platform];
  const channelKeys = Object.keys(p.channels);

  const activeChannel = selectedChannel && p.channels[selectedChannel] ? selectedChannel : null;
  const channelInfo = activeChannel ? p.channels[activeChannel] : null;

  const isDV360Genre = platform === "DV360" && channelInfo && channelInfo.status === "genre";
  const isDV360Topics = platform === "DV360" && channelInfo && channelInfo.status === "topics";
  const isDV360Full = platform === "DV360" && channelInfo && channelInfo.status === "full";
  const showIAB = platform !== "DV360" || isDV360Full;

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    if (!q) return iabCategories;
    return iabCategories.map(c => {
      const nameMatch = c.name.toLowerCase().includes(q) || c.id.toLowerCase().includes(q);
      const filteredSubs = c.subs.filter(s => s.toLowerCase().includes(q));
      if (nameMatch || filteredSubs.length > 0) return { ...c, subs: nameMatch ? c.subs : filteredSubs, _highlight: true };
      return null;
    }).filter(Boolean);
  }, [search]);

  const totalSubs = iabCategories.reduce((a, c) => a + c.subs.length, 0);

  return (
    <div style={{ fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", maxWidth: 900, margin: "0 auto", color: "#1a1a2e" }}>
      {/* Header */}
      <div style={{ background: "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)", borderRadius: 16, padding: "28px 32px", marginBottom: 24, color: "#fff" }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>IAB Content Categories Explorer</h1>
        <p style={{ margin: "6px 0 0", opacity: 0.7, fontSize: 13 }}>Channel-wise category availability across Yahoo DSP, The Trade Desk & DV360</p>
      </div>

      {/* Platform Tabs */}
      <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
        {Object.entries(data).map(([name, d]) => (
          <button key={name} onClick={() => { setPlatform(name); setSelectedChannel(null); setExpandedCat(null); setSearch(""); setTab("categories"); }}
            style={{
              flex: 1, minWidth: 140, padding: "14px 12px", border: platform === name ? `2px solid ${d.color}` : "2px solid #e5e7eb",
              borderRadius: 12, background: platform === name ? `${d.color}10` : "#fff",
              cursor: "pointer", transition: "all .2s", fontWeight: 600, fontSize: 14, color: platform === name ? d.color : "#6b7280"
            }}>
            {name}
          </button>
        ))}
      </div>

      {/* Platform Info */}
      <div style={{ background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: 12, padding: "14px 20px", marginBottom: 20, display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
        <div>
          <span style={{ fontSize: 12, color: "#64748b", textTransform: "uppercase", letterSpacing: 1 }}>Taxonomy</span>
          <div style={{ fontWeight: 600, fontSize: 14, marginTop: 2 }}>{p.taxonomy}</div>
        </div>
        <div style={{ textAlign: "right" }}>
          <span style={{ fontSize: 12, color: "#64748b", textTransform: "uppercase", letterSpacing: 1 }}>Channels</span>
          <div style={{ fontWeight: 600, fontSize: 14, marginTop: 2 }}>{channelKeys.length} available</div>
        </div>
      </div>

      {/* Channel Cards */}
      <div style={{ marginBottom: 24 }}>
        <h3 style={{ fontSize: 14, fontWeight: 600, color: "#475569", marginBottom: 10, textTransform: "uppercase", letterSpacing: 1 }}>Select Channel</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 10 }}>
          {channelKeys.map(ch => {
            const c = p.channels[ch];
            const sel = activeChannel === ch;
            return (
              <button key={ch} onClick={() => { setSelectedChannel(sel ? null : ch); setTab("categories"); }}
                style={{
                  padding: "12px 14px", border: sel ? `2px solid ${p.color}` : "1px solid #e5e7eb",
                  borderRadius: 10, background: sel ? `${p.color}08` : "#fff", cursor: "pointer",
                  textAlign: "left", transition: "all .15s"
                }}>
                <div style={{ fontSize: 18, marginBottom: 4 }}>{c.icon}</div>
                <div style={{ fontSize: 13, fontWeight: 600, color: "#1e293b", lineHeight: 1.3 }}>{ch}</div>
                <div style={{ marginTop: 6 }}><Badge type={c.status} /></div>
                {c.note && <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 4 }}>{c.note}</div>}
              </button>
            );
          })}
        </div>
      </div>

      {/* Content Area */}
      {activeChannel && (
        <div style={{ border: "1px solid #e2e8f0", borderRadius: 14, overflow: "hidden" }}>
          {/* Tab bar */}
          <div style={{ display: "flex", background: "#f1f5f9", borderBottom: "1px solid #e2e8f0" }}>
            {["categories", "sensitive"].map(t => (
              <button key={t} onClick={() => setTab(t)}
                style={{
                  flex: 1, padding: "12px 16px", border: "none", cursor: "pointer", fontWeight: 600, fontSize: 13,
                  background: tab === t ? "#fff" : "transparent", color: tab === t ? p.color : "#64748b",
                  borderBottom: tab === t ? `2px solid ${p.color}` : "2px solid transparent"
                }}>
                {t === "categories" ? `📋 Content Categories` : `🛡️ Brand Safety Exclusions`}
              </button>
            ))}
          </div>

          {tab === "categories" && (
            <div style={{ padding: 20 }}>
              {/* Header info */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, flexWrap: "wrap", gap: 8 }}>
                <div>
                  <span style={{ fontSize: 16, fontWeight: 700 }}>{activeChannel}</span>
                  <span style={{ fontSize: 12, color: "#94a3b8", marginLeft: 8 }}>
                    {showIAB ? `${iabCategories.length} Tier 1 · ${totalSubs} Tier 2` : isDV360Genre ? "13 Genres" : isDV360Topics ? "2,000+ Topics" : ""}
                  </span>
                </div>
                {showIAB && (
                  <input placeholder="Search categories..." value={search} onChange={e => setSearch(e.target.value)}
                    style={{ padding: "8px 14px", border: "1px solid #d1d5db", borderRadius: 8, fontSize: 13, width: 220, outline: "none" }} />
                )}
              </div>

              {/* IAB Categories */}
              {showIAB && filtered.map(cat => (
                <div key={cat.id} style={{ marginBottom: 4 }}>
                  <button onClick={() => setExpandedCat(expandedCat === cat.id ? null : cat.id)}
                    style={{
                      width: "100%", display: "flex", justifyContent: "space-between", alignItems: "center",
                      padding: "10px 14px", border: "1px solid #e5e7eb", borderRadius: 8, cursor: "pointer",
                      background: expandedCat === cat.id ? `${p.color}08` : "#fafafa", transition: "all .15s"
                    }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <span style={{ fontFamily: "monospace", fontSize: 11, background: "#e2e8f0", padding: "2px 6px", borderRadius: 4, color: "#475569" }}>{cat.id}</span>
                      <span style={{ fontWeight: 600, fontSize: 13 }}>{cat.name}</span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontSize: 11, color: "#94a3b8" }}>{cat.subs.length} sub{cat.subs.length !== 1 ? "s" : ""}</span>
                      <span style={{ fontSize: 12, transform: expandedCat === cat.id ? "rotate(180deg)" : "rotate(0deg)", transition: "transform .2s" }}>▼</span>
                    </div>
                  </button>
                  {expandedCat === cat.id && cat.subs.length > 0 && (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, padding: "10px 14px", background: "#f8fafc", borderRadius: "0 0 8px 8px", border: "1px solid #e5e7eb", borderTop: "none" }}>
                      {cat.subs.map((s, i) => (
                        <span key={i} style={{ background: "#fff", border: "1px solid #e2e8f0", padding: "4px 10px", borderRadius: 6, fontSize: 12, color: "#334155" }}>
                          {cat.id}-{i + 1} {s}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              {showIAB && filtered.length === 0 && <p style={{ textAlign: "center", color: "#94a3b8", padding: 20 }}>No categories match your search.</p>}

              {/* DV360 Content Categories */}
              {isDV360Full && !showIAB && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                  {dv360Categories.map((c, i) => (
                    <span key={i} style={{ background: "#EFF6FF", border: "1px solid #BFDBFE", padding: "8px 14px", borderRadius: 8, fontSize: 13, fontWeight: 500, color: "#1E40AF" }}>/{c}</span>
                  ))}
                </div>
              )}

              {/* DV360 Genre */}
              {isDV360Genre && (
                <div>
                  <p style={{ fontSize: 13, color: "#64748b", marginBottom: 12 }}>CTV and Audio line items use a simplified Genre targeting system instead of the full category taxonomy.</p>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                    {dv360Genres.map((g, i) => (
                      <span key={i} style={{ background: "#EFF6FF", border: "1px solid #BFDBFE", padding: "10px 16px", borderRadius: 8, fontSize: 13, fontWeight: 600, color: "#1E40AF" }}>/{g}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* DV360 Topics */}
              {isDV360Topics && (
                <div>
                  <p style={{ fontSize: 13, color: "#64748b", marginBottom: 12 }}>YouTube TrueView uses Google's Topics taxonomy with 2,000+ topics. Top-level topics include:</p>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                    {[...dv360Categories.filter(c => c !== "Sensitive Subjects" && c !== "World Localities" && c !== "Reference" && c !== "Online Communities"), "Apparel", "Beauty & Personal Care", "Occasions & Gifts"].map((g, i) => (
                      <span key={i} style={{ background: "#F3E8FF", border: "1px solid #D8B4FE", padding: "10px 16px", borderRadius: 8, fontSize: 13, fontWeight: 600, color: "#6B21A8" }}>/{g}</span>
                    ))}
                  </div>
                  <p style={{ fontSize: 12, color: "#94a3b8", marginTop: 12 }}>Each top-level topic contains deep subcategories. Full list available in DV360 UI under line item targeting.</p>
                </div>
              )}
            </div>
          )}

          {tab === "sensitive" && (
            <div style={{ padding: 20 }}>
              <p style={{ fontSize: 13, color: "#64748b", marginBottom: 16 }}>Categories available for exclusion to protect brand safety on <strong>{platform}</strong>:</p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {sensitiveCategories[platform].map((c, i) => (
                  <span key={i} style={{ background: "#FEF2F2", border: "1px solid #FECACA", padding: "8px 14px", borderRadius: 8, fontSize: 13, fontWeight: 500, color: "#991B1B", display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ fontSize: 10 }}>🚫</span> {c}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {!activeChannel && (
        <div style={{ textAlign: "center", padding: "40px 20px", color: "#94a3b8", border: "2px dashed #e2e8f0", borderRadius: 14 }}>
          <div style={{ fontSize: 32, marginBottom: 8 }}>👆</div>
          <div style={{ fontSize: 14, fontWeight: 500 }}>Select a channel above to view available IAB categories</div>
        </div>
      )}

      {/* Footer */}
      <div style={{ marginTop: 24, padding: "14px 20px", background: "#f8fafc", borderRadius: 10, fontSize: 11, color: "#94a3b8", lineHeight: 1.6 }}>
        <strong>Notes:</strong> Yahoo DSP & TTD use the standard IAB v2.x taxonomy across all channels. TTD additionally supports IAB v3.0 (1,500+ categories). DV360 uses Google's own category system — full categories for Display/Video, Genre subset for CTV/Audio, and Topics for YouTube. Always verify live lists within each platform's UI or API. Last updated April 2026.
      </div>
    </div>
  );
}
