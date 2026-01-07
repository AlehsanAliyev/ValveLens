import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";

import "./styles.css";
import Devices from "./pages/Devices";
import Home from "./pages/Home";
import Live from "./pages/Live";
import Zones from "./pages/Zones";

function App() {
  return (
    <div className="app">
      <nav className="nav">
        <div className="nav-title">ValveLens</div>
        <div className="nav-links">
          <Link to="/">Home</Link>
          <Link to="/live/webcam">Live</Link>
          <Link to="/zones">Zones</Link>
          <Link to="/devices">Devices</Link>
        </div>
      </nav>
      <main className="content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/live/:mode" element={<Live />} />
          <Route path="/zones" element={<Zones />} />
          <Route path="/devices" element={<Devices />} />
        </Routes>
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
