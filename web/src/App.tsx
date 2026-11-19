import { NavLink, Route, Routes } from "react-router-dom";
import AqiMap from "./pages/AqiMap";
import HchoHotspots from "./pages/HchoHotspots";
import FireTransport from "./pages/FireTransport";
import Validation from "./pages/Validation";
import About from "./pages/About";

const links = [
  { to: "/", label: "AQI Map" },
  { to: "/hcho", label: "HCHO Hotspots" },
  { to: "/fire", label: "Fire & Transport" },
  { to: "/validation", label: "Validation" },
  { to: "/about", label: "About" },
];

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <nav className="flex gap-4 p-3 border-b">
        {links.map((l) => (
          <NavLink key={l.to} to={l.to} end className={({ isActive }) => (isActive ? "font-semibold" : "")}>
            {l.label}
          </NavLink>
        ))}
      </nav>
      <main className="flex-1 p-4">
        <Routes>
          <Route path="/" element={<AqiMap />} />
          <Route path="/hcho" element={<HchoHotspots />} />
          <Route path="/fire" element={<FireTransport />} />
          <Route path="/validation" element={<Validation />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
    </div>
  );
}
